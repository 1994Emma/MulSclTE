import numpy as np
from scipy import spatial

from change_pointer_detector import ChangePointDetector
from dataset_statistc_info import (
    mean_gt_action_classes_datasets,
    mean_gt_action_num_datasets,
    max_gt_activity_action_classes_num_datasets,
    mean_gt_activity_action_classes_num_datasets,
    mean_gt_activity_action_num_datasets,
)

from myutil import np_load_gt, _utils_basic_logger, np_load_feature


####################################################
# Basic Util Functions
####################################################
def generate_gt_boundary_segmentations_for_each_video(gt_path):
    """
    Generate frame-level segments from ground-truth labels.
    Each segment is defined by a label, start index, and end index.

    Args:
        gt_path (str): Path to the ground truth label file.

    Returns:
        tuple:
            - y_gt_action (np.ndarray): original frame-wise label array.
            - gt_seg_frames (List[List[int]]): list of segments [label, start, end].
    """
    y_gt_action = np_load_gt(gt_path)
    gt_seg_frames = []

    c_label = y_gt_action[0]
    c_start_idx = 0

    for idx in range(1, len(y_gt_action)):
        if y_gt_action[idx] != c_label:
            gt_seg_frames.append([c_label, c_start_idx, idx - 1])
            c_label = y_gt_action[idx]
            c_start_idx = idx

    gt_seg_frames.append([c_label, c_start_idx, len(y_gt_action) - 1])
    return y_gt_action, gt_seg_frames


def generate_gt_frame_seg_label(gt_seg_frames):
    """
    Generate pseudo-class IDs per segment, converting segments to frame-wise labels.

    Args:
        gt_seg_frames (List[List[int]]): list of segments [label, start, end]

    Returns:
        tuple:
            - gt_frames (List[int]): segment index per frame.
            - gt_class_no (int): number of unique pseudo-classes.
    """
    gt_frames = []
    for i, seg in enumerate(gt_seg_frames):
        gt_frames.extend([i] * (seg[2] - seg[1] + 1))
    return gt_frames, len(gt_seg_frames)


def load_x(feature_path, feature_dim):
    """
    Load frame-wise feature sequence and ensure correct shape (seq_len, dim).

    Args:
        feature_path (str): Feature file path.
        feature_dim (int): Expected dimensionality of features.

    Returns:
        np.ndarray: Frame-wise features.
    """
    _utils_basic_logger.debug(f"call load_x({feature_path}, {feature_dim})")
    features = np_load_feature(feature_path)

    if features.shape[0] == feature_dim:
        features = features.T

    _utils_basic_logger.debug(f"features.shape={features.shape}")
    return features


def load_y(gt_path):
    """
    Load ground-truth and generate segment-level labels (ignoring class identity).

    Args:
        gt_path (str): Path to ground truth.

    Returns:
        tuple: y_gt_action, gt_seg_frames, gt_frames_ignore_class, gt_class_no
    """
    _utils_basic_logger.debug(f"call load_y({gt_path})")
    y_gt_action, gt_seg_frames = generate_gt_boundary_segmentations_for_each_video(gt_path)
    gt_frames_ignore_class, gt_class_no = generate_gt_frame_seg_label(gt_seg_frames)

    return y_gt_action, gt_seg_frames, gt_frames_ignore_class, gt_class_no


####################################################
# Segmenter Utility
####################################################
def fuse_sequences(seq1, seq2, w1, w2, normalize=False):
    """
    Linearly combine two sequences, optionally normalize.
    """
    min_len = min(len(seq1), len(seq2))
    seq1 = np.array(seq1[:min_len])
    seq2 = np.array(seq2[:min_len])

    if normalize:
        seq1 = (seq1 - np.min(seq1)) / (np.ptp(seq1) + 1e-8)
        seq2 = (seq2 - np.min(seq2)) / (np.ptp(seq2) + 1e-8)

    return w1 * seq1 + w2 * seq2


def cal_cosine_similarities(feature_path, feature_dim):
    """
    Compute cosine similarity between each pair of adjacent frames.

    Returns:
        List[float]: similarity sequence of length (n_frames - 1)
    """
    features = load_x(feature_path, feature_dim)
    similarities = [
        1 - spatial.distance.cosine(features[i], features[i + 1])
        for i in range(len(features) - 1)
    ]
    return similarities


def detect_change_points(seq, dcp_method, win_size=None, n_bkps=None, min_size=None, jump=None):
    """
    Change point detection wrapper.
    """
    if dcp_method == "win_search":
        return ChangePointDetector.detect_change_points(
            -1 * np.array(seq), dcp_method, win_size, n_bkps, min_size, jump
        )
    elif dcp_method == "argrelextrema":
        return ChangePointDetector.detect_change_points(
            np.array(seq), dcp_method, win_size=win_size
        )
    else:
        raise ValueError(f"Unsupported dcp_method: {dcp_method}")


def convert_boundary_points_to_frame_labels(boundary_points):
    """
    Convert boundary indices to frame-wise segment labels.

    Args:
        boundary_points (List[int]): list of change points, including first and last frame.

    Returns:
        Tuple[List[int], int]: frame labels, number of segments
    """
    pred_labels = []
    start_idx = 0
    class_no = 0
    frame_num = boundary_points[-1] + 1

    for end_idx in boundary_points[1:]:
        pred_labels.extend([class_no] * (end_idx - start_idx))
        class_no += 1
        start_idx = end_idx

    while len(pred_labels) < frame_num:
        pred_labels.append(class_no - 1)

    return pred_labels, class_no


##################################################
# Adaptive K Selection
##################################################
def get_valid_k_types():
    """
    Return the valid types of K configurations.
    """
    full_k_types = ["mean_classes", "mean_actions"]
    activity_k_types = ["mean_activity_classes", "mean_activity_actions", "max_activity_classes"]
    return full_k_types + activity_k_types, full_k_types, activity_k_types


def get_K(k_type, ds_name=None, activity_name=None):
    """
    Determine number of clusters (K) based on dataset or activity statistics.
    """
    valid_k_types, full_k_types, activity_k_types = get_valid_k_types()
    assert k_type in valid_k_types, f"Illegal k_type='{k_type}', should be in {valid_k_types}"

    if k_type in full_k_types:
        assert ds_name is not None, "ds_name is required for global K estimation"
        if k_type == "mean_classes":
            return mean_gt_action_classes_datasets[ds_name]
        else:
            return mean_gt_action_num_datasets[ds_name]
    else:
        assert ds_name is not None and activity_name is not None, "Both ds_name and activity_name are required"
        if k_type == "mean_activity_classes":
            return mean_gt_activity_action_classes_num_datasets[ds_name][activity_name]
        elif k_type == "mean_activity_actions":
            return mean_gt_activity_action_num_datasets[ds_name][activity_name]
        elif k_type == "max_activity_classes":
            return max_gt_activity_action_classes_num_datasets[ds_name][activity_name]
        else:
            raise ValueError(f"Unsupported k_type: {k_type}")
