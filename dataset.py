from torch.utils.data import Dataset, DataLoader
import torch
import os
import glob
import numpy as np

from myutil import np_load_gt, np_load_feature, _utils_basic_logger


def generate_label2action_dict(src_mapping):
    label2action_mappings = {}
    for row in src_mapping:
        label2action_mappings[int(row[0])] = row[1]
    return label2action_mappings


def generate_action2label_dict(src_mapping):
    action2label_mappings = {}
    for row in src_mapping:
        action2label_mappings[row[1]] = int(row[0])
    return action2label_mappings


def labels2actions(labels, label2action_mappings):
    actions = [label2action_mappings[label] for label in labels]
    return np.array(actions)


def actions2labels(actions, action2label_mappings):
    labels = [action2label_mappings[action] for action in actions]
    return np.array(labels)


def get_clip_indexes_by_slide_windows(seq_len, window_size, step):
    """
    Generate indices for video clips using sliding windows for training.

    :param seq_len: Length of the sequence (video).
    :param window_size: Size of the sliding window for each clip.
    :param step: Step size for sliding the window.
    :return: List of clip start and end indices.
    """
    indexes = []
    start_idx = 0

    if seq_len <= window_size:
        # if the sequence length is less than the window size
        indexes = [[0, seq_len]]
        return indexes

    if (seq_len > window_size) and (seq_len < window_size + step):
        # Adjust step size
        step = seq_len - window_size

    # Generate clip indices with sliding windows
    while start_idx <= (seq_len - window_size):
        indexes.append([start_idx, start_idx + window_size])
        start_idx = start_idx + step

    # Handle the last clip if there is remaining sequence length
    if (start_idx > (seq_len - window_size)) and (start_idx < seq_len) and (
            indexes[-1][1] - indexes[-1][0] < (window_size)):
        indexes.append([start_idx, seq_len])

    return indexes


class FeatureDataset(Dataset):
    def __init__(self, root_dir, feature_type="i3d", using_clip=False, args=None,
                 debug_phase=False, clip_split_method="train_split"):
        """
        Initializes the dataset for feature extraction and ground truth.

        :param root_dir: Root directory containing feature and ground truth files.
        :param feature_type: Type of features.
        :param using_clip: Whether to use clip-based data or whole-video based data.
        :param args: Arguments containing additional parameters.
        :param debug_phase: Flag to enable debug mode with a small set sampled from Breakfast.
        :param clip_split_method: Method to generate clips ("train_split", ).
        """

        self.args = args
        self.using_clip = using_clip
        self.ds_name = self.args.ds_name
        self.feature_type = feature_type
        self.clip_split_method = clip_split_method
        self.root_dir = root_dir

        _utils_basic_logger.info("feature_type={}".format(feature_type))

        # Set feature and ground truth directories based on dataset name and debug phase
        self.cur_feature_dir, self.gt_dir = self._get_feature_and_gt_dirs(debug_phase)
        self.mapping_path, self.mapping_eval_path = self._get_mapping_paths()

        # Generate label2action and action2label mappings
        self.mappings = np_load_gt(self.mapping_path)
        self.label2action_mappings = generate_label2action_dict(self.mappings)
        self.action2label_mappings = generate_action2label_dict(self.mappings)

        # Get file paths for features and ground truth
        if "YTI" in self.ds_name:
            gt_filenames = [fn for fn in glob.glob(os.path.join(self.gt_dir, "**", "*"), recursive=True) if
                            os.path.isfile(fn)]
            gt_filenames = [fn for fn in gt_filenames if "idt" not in fn]
            self.gt_filenames = {os.path.split(fn)[-1].split(".")[0]: fn for fn in gt_filenames}
            feature_paths = [fn for fn in glob.glob(os.path.join(self.cur_feature_dir, "**", "*.txt"), recursive=True)
                             if os.path.isfile(fn)]
            self.feature_filenames = {os.path.split(fn)[-1].split(".")[0]: fn for fn in feature_paths}
        else:
            self.gt_filenames = {os.path.split(fn)[-1].split(".")[0]: fn for fn in
                                 glob.glob(os.path.join(self.gt_dir, "**", "*"), recursive=True) if os.path.isfile(fn)}

            # get feature paths
            self.feature_filenames = {os.path.split(fn)[-1].split(".")[0]: fn for fn in
                                      glob.glob(os.path.join(self.cur_feature_dir, "**", "*"), recursive=True) if os.path.isfile(fn)}

        _utils_basic_logger.debug(
            "len(feature_filenames.keys())={}, feature_filenames.keys()={}".format(len(self.feature_filenames.keys()),
                                                                                   self.feature_filenames.keys()))

        _utils_basic_logger.debug(
            "len(gt_filenames.keys())={}, gt_filenames.keys()={}".format(len(self.gt_filenames.keys()),
                                                                         self.gt_filenames.keys()))

        assert len(self.feature_filenames.keys()) == len(
            self.gt_filenames.keys()), "The numbers of feature and gt files are not equal"

        assert set(self.feature_filenames.keys()) == set(
            self.gt_filenames.keys()), "The filenames of features and gt are not equal"

        # Generate feature-ground truth paths
        self.feature_gt_paths = self.get_feature_gt_paths()

    def _get_feature_and_gt_dirs(self, debug_phase):
        # Define feature and ground truth directories for each dataset
        dataset_configs = {
            "Breakfast": {
                "feature_dir": "I3D_part" if debug_phase else "I3D_2048_features",
                "gt_dir": "gt_part" if debug_phase else "groundTruth"
            },
            "50Salads": {
                "feature_dir": "features",
                "gt_dir": "groundTruth"
            },
            "default": {  # For YTI and EPIC-KITCHENS
                "feature_dir": "features",
                "gt_dir": "groundTruth"
            }
        }

        config = dataset_configs.get(self.ds_name, dataset_configs["default"])
        return (
            os.path.join(self.root_dir, config["feature_dir"]),
            os.path.join(self.root_dir, config["gt_dir"])
        )

    def _get_mapping_paths(self):
        # Define mapping paths for each dataset
        dataset_configs = {
            "YTI": {
                "mapping_path": os.path.join(self.root_dir, "mapping", "mapping.txt"),
                "mapping_eval_path": None
            },
            "50Salads": {
                "mapping_path": os.path.join(self.root_dir, "mapping", "mapping.txt"),
                "mapping_eval_path": os.path.join(self.root_dir, "mapping", "mappingeval.txt")
            },
            "default": {  # For Breakfast and EPIC-KITCHENS
                "mapping_path": os.path.join(self.root_dir, "mapping.txt"),
                "mapping_eval_path": None
            }
        }

        config = dataset_configs.get(self.ds_name, dataset_configs["default"])
        return config["mapping_path"], config["mapping_eval_path"]

    def get_feature_gt_paths(self):
        feature_gt_paths = []
        keys = sorted(list(self.gt_filenames.keys()))
        for key in keys:
            c_feature_path, c_gt_path = self.feature_filenames[key], self.gt_filenames[key]
            c_y_data = np_load_gt(c_gt_path)

            c_frame_len = len(c_y_data)

            # [start_idx, end_idx)
            indexes = [[0, c_frame_len]]

            # if using clip
            if self.using_clip:
                if self.clip_split_method == "train_split":
                    indexes = get_clip_indexes_by_slide_windows(c_frame_len, self.args.clip_window_size,
                                                                self.args.clip_window_step)

            for index in indexes:
                feature_gt_paths.append([c_feature_path, c_gt_path, index[0], index[1]])

        return feature_gt_paths

    def __len__(self):
        return len(self.feature_gt_paths)

    def __getitem__(self, idx):
        """
        Get a data sample from the dataset.
        :param idx: Index of the sample.
        :return: A tuple containing feature data, ground truth data, evaluation data, and metadata.
        """
        c_feature_path, c_gt_path, start_idx, end_idx = self.feature_gt_paths[idx]

        x_data = np_load_feature(c_feature_path)

        if (self.feature_type == "i3d") and (self.ds_name in ["Breakfast", "50Salads"]):
            x_data = x_data.T

        y_src_data = np_load_gt(c_gt_path)
        y_data = actions2labels(y_src_data, self.action2label_mappings)

        if "YTI" in self.ds_name:
            # Ensure no negative numbers: fix the error only on YTI
            gt_min = np.min(y_data)
            if gt_min < 0:
                y_data = y_data - gt_min

        # only for 50Salads
        y_data_eval = []
        if self.ds_name == "50Salads":
            y_data_eval = np.array([self.action2label_mappings_eval[self.label2action_mappings[val]] for val in y_data])

        # using clip
        if self.using_clip:
            if (end_idx - start_idx - 1) < self.args.clip_window_size:
                zero_padding = True
            else:
                zero_padding = False

            x_data = x_data[start_idx:end_idx, :]
            y_data = y_data[start_idx:end_idx, ]

            if self.args.zero_padding and zero_padding:
                new_x_data = np.zeros(shape=(self.args.clip_window_size, x_data.shape[1]), dtype="float32")
                new_y_data = -1 * np.ones(shape=(self.args.clip_window_size,))
                new_x_data[:x_data.shape[0], :] = x_data
                new_y_data[:y_data.shape[0]] = y_data
                x_data = new_x_data
                y_data = new_y_data

            if self.ds_name == "50Salads":
                y_data_eval = y_data_eval[start_idx:end_idx, ]

        return x_data, y_data, y_data_eval, c_feature_path, c_gt_path, start_idx, end_idx


def get_dataloader(data_root, feature_type, batch_size, using_clip, shuffle=True, args=None,
                   clip_split_method="train_split"):
    dataset = FeatureDataset(data_root, feature_type=feature_type, using_clip=using_clip, args=args,
                             debug_phase=args.debug_phase, clip_split_method=clip_split_method)

    print("dataset length={}".format(dataset.__len__()))
    my_generator = torch.Generator(device=args.device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=my_generator)

    return dataloader