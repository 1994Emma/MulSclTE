import abc
import os
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
from sklearn import metrics

from myutil import _utils_basic_logger, make_dirs, Averager
from dataset import FeatureDataset, actions2labels
from segmenter_util import (
    fuse_sequences, load_y, cal_cosine_similarities, detect_change_points,
    convert_boundary_points_to_frame_labels, load_x, get_K
)
from finch.finch_for_refining_pred_loss import FINCH_AS_REFINER


####################################################
# Hungarian Matching
####################################################
class Hungarian:
    @classmethod
    def matching(cls, gt_labels, pred_labels):
        """
        Align predicted labels with ground-truth labels using Hungarian algorithm.
        Returns reordered predictions for optimal alignment.
        """
        cost_matrix = cls.estimate_cost_matrix(gt_labels, pred_labels)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        return col_ind[pred_labels]

    @classmethod
    def estimate_cost_matrix(cls, gt_labels, cluster_labels):
        """
        Construct cost matrix (negated confusion matrix) for Hungarian matching.
        """
        assert len(gt_labels) == len(cluster_labels)
        unique_gt = np.unique(gt_labels)
        unique_pred = np.unique(cluster_labels)
        n_pred = len(unique_pred)
        matrix_dim = max(n_pred, np.max(unique_gt) + 1)

        cost_matrix = np.zeros((n_pred, matrix_dim))
        for i in unique_pred:
            indices = np.where(cluster_labels == i)
            selected_gt = gt_labels[indices]
            for j in unique_gt:
                cost_matrix[i][j] = np.sum(selected_gt == j)
        return -cost_matrix


####################################################
# PSSeg Segmenter
####################################################
class Refiner:
    """
    Post-processing segment refiner that supports k-means and FINCH clustering.
    """
    valid_refine_methods = ["kmeans", "finch", "tw-finch"]
    valid_finch_types = ["s", "tw"]  # "s" = standard FINCH, "tw" = temporal window-based FINCH

    @classmethod
    def get_segment_features(cls, x_data, change_points):
        """
        Compute mean feature vector for each coarse segment.
        """
        start_idx = 0
        seg_features, init_partitions = [], []

        for i, c_point in enumerate(change_points):
            if i == 0:
                continue
            elif i == len(change_points) - 1:
                c_feature = np.mean(x_data[start_idx: c_point + 1, :], axis=0)
            else:
                c_feature = np.mean(x_data[start_idx: c_point, :], axis=0)

            seg_features.append(c_feature)
            init_partitions.append([start_idx, c_point])
            start_idx = c_point

        return seg_features, init_partitions

    @classmethod
    def refine_by_kmeans(cls, x_data, initial_change_points, n_clusters=None):
        """
        Use k-means to refine segments based on average segment features.
        """
        seg_features, init_partitions = cls.get_segment_features(x_data, initial_change_points)
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(seg_features)
        pre_seg_labels = kmeans.labels_

        refined_labels = []
        for idx, c_label in enumerate(pre_seg_labels):
            start_idx = initial_change_points[idx]
            end_idx = initial_change_points[idx + 1]
            if idx == len(pre_seg_labels) - 1:
                end_idx += 1
            refined_labels.extend([c_label] * (end_idx - start_idx))

        return refined_labels

    @classmethod
    def refine_by_finches(cls, x_data, initial_change_points, n_clusters, finch_type):
        """
        Refine segments using FINCH clustering.
        """
        assert finch_type in cls.valid_finch_types
        init_pred_labels, init_pred_class_no = convert_boundary_points_to_frame_labels(initial_change_points)

        # Apply FINCH-based refinement
        _, _, refined_pred = FINCH_AS_REFINER(
            x_data,
            req_clust=n_clusters,
            finch_type=finch_type,
            verbose=False,
            init_group=init_pred_labels,
            init_num_clust=init_pred_class_no
        )
        return refined_pred

    @classmethod
    def refinement(cls, refine_method, x_data, initial_change_points, n_clusters):
        """
        Main entry to refine segments using the specified clustering method.
        """
        assert refine_method in cls.valid_refine_methods

        if refine_method == "kmeans":
            return cls.refine_by_kmeans(x_data, initial_change_points, n_clusters)
        elif refine_method == "finch":
            return cls.refine_by_finches(x_data, initial_change_points, n_clusters, finch_type="s")
        elif refine_method == "tw-finch":
            return cls.refine_by_finches(x_data, initial_change_points, n_clusters, finch_type="tw")


default_ps_seg_params = {
    "dcp": "argrelextrema",       # Change point detection method
    "refine": "finch",            # Refinement clustering method
    "k_type": "mean_classes",     # Strategy to determine cluster count
    "w_size": 140,                # Smoothing window size
    "fuse_weights": [0.7, 0.3]    # Fusion weights [prediction error, similarity]
}


class PSSeg:
    """
    PS-Seg main implementation. Uses prediction loss and similarity for segmentation.
    """

    def __init__(self, pred_loss_csv_path, data_root, feature_root, feature_dim,
                 save_root="./", ps_seg_params=default_ps_seg_params, args=None):

        super().__init__()

        self.pred_loss_path = pred_loss_csv_path
        self.data_root = data_root
        self.ds_name = args.ds_name
        self.feature_dim = feature_dim
        self.feature_root = feature_root
        self.dataset = FeatureDataset(self.data_root, args=args)

        self.save_root = save_root
        make_dirs(self.save_root)
        self.output_seg_path = os.path.join(self.save_root, "./ps_seg_result.csv")

        self.params = ps_seg_params
        self.args = args

    def segment(self, max_video_num=None):
        """
        Core segmentation loop.
        For each video: load error/similarity → fusion → detect boundaries → refine → decode.
        """
        df_output = []
        # load prediction error sequence
        df_losses = pd.read_csv(self.pred_loss_path)

        if max_video_num is not None:
            df_losses = df_losses.iloc[:max_video_num]

        for idx, row in tqdm(df_losses.iterrows(), total=len(df_losses), desc="Processing videos"):
            feature_path = eval(row["feature_path"])[0]
            gt_path = eval(row["gt_path"])[0]
            start_idx, end_idx = row["start_idx"], row["end_idx"]
            pred_losses_seq = eval(row['loss_each'])

            # Extract activity name for class-count estimation
            if self.ds_name == "Breakfast":
                activity_name = os.path.basename(feature_path).split(".")[0].split("_")[-1]
            elif "YTI" in self.ds_name:
                activity_name = "_".join(os.path.basename(gt_path).split("_")[:-1])
            else:
                activity_name = "rgb"

            # Step 1: Load ground-truth labels
            gt_data, _, _, _ = load_y(gt_path)
            gt_labels = actions2labels(gt_data, self.dataset.action2label_mappings)
            frame_num = len(gt_data)

            # If the dataset is YTI and there are negative labels in gt_labels,
            # shift all labels so that the minimum value becomes 0
            if "YTI" in self.ds_name and np.min(gt_labels) < 0:
                # Ensure all labels are non-negative
                gt_labels = gt_labels - np.min(gt_labels)

            # Step 2: Calculate frame-wise cosine similarity
            filename = os.path.basename(feature_path)
            real_feature_path = os.path.join(self.feature_root, filename)
            similarity_seq = cal_cosine_similarities(real_feature_path, self.feature_dim)

            # Step 3: Fuse prediction error and similarity
            pred_w, sim_w = self.params["fuse_weights"]
            fusion_seq = fuse_sequences(pred_losses_seq, -1.0 * np.array(similarity_seq), pred_w, sim_w, True)

            # Step 4: Detect change points
            K = get_K(self.params["k_type"], self.ds_name, activity_name)
            win_size = self.params["w_size"]
            n_bkps = K * 5
            min_size, jump = 2, 5
            initial_change_points = detect_change_points(fusion_seq, self.params["dcp"], win_size, n_bkps, min_size,
                                                         jump)
            initial_change_points = [0] + initial_change_points + [frame_num - 1]

            # Step 5: Refine segments using clustering
            x_data = load_x(real_feature_path, self.feature_dim)
            refined_pred_labels = Refiner.refinement(self.params["refine"], x_data, initial_change_points, K)

            # Step 6: Hungarian decode for label alignment
            decode_y_pred = Hungarian.matching(gt_labels, refined_pred_labels)

            # Save result
            df_output.append([
                feature_path, gt_path, gt_labels,
                initial_change_points, refined_pred_labels, decode_y_pred
            ])

        df_output = pd.DataFrame(df_output, columns=[
            "feature_path", "gt_path", "gt_labels", "initial_cpoints",
            "refined_pred_labels", "decode_pred_labels"
        ])
        df_output.to_csv(self.output_seg_path, index=False)

        _utils_basic_logger.info(f"Writing results into: {self.output_seg_path}")

        return df_output

