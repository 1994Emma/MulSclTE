import argparse
from myutil import seed_everything, pprint
from segmenter import PSSeg


def parse_args():
    parser = argparse.ArgumentParser(description="Unsupervised Action Segmentation using PS-Seg")

    # Dataset configuration
    parser.add_argument('--ds_name', type=str, default="Breakfast",
                        help="Dataset name. Examples: Breakfast, 50Salads, YTI, etc.")
    parser.add_argument('--data_root', type=str, required=True,
                        help="Root path of the original dataset.")
    parser.add_argument('--salads_eval_level', action='store_true',
                        help="Use action-level granularity evaluation for 50Salads dataset.")

    # Prediction loss CSV
    parser.add_argument('--pred_loss_csv_path', type=str, required=True,
                        help="CSV file storing frame-level prediction loss for all videos.")

    # Feature configuration
    parser.add_argument('--feature_root', type=str, required=True,
                        help="Root directory of extracted features.")
    parser.add_argument('--feature_dim', type=int, required=True,
                        help="Dimensionality of the extracted feature vectors.")

    # Output path
    parser.add_argument('--save_root', type=str, default="./outputs",
                        help="Directory to save output results.")

    # PS-Seg parameters
    parser.add_argument('--dcp', type=str, default="argrelextrema",
                        choices=["argrelextrema"],
                        help="Method to detect candidate change points.")

    parser.add_argument('--refine', type=str, default="finch",
                        choices=["kmeans", "finch", "tw-finch"],
                        help="Clustering method used in the refinement stage.")

    parser.add_argument('--k_type', type=str, default="mean_classes",
                        choices=["mean_classes", "mean_activity_classes"],
                        help="Strategy to determine the number of clusters.")

    parser.add_argument('--w_size', type=int, default=120,
                        help="Window size for smoothing. Set -1 for automatic selection.")

    parser.add_argument('--fuse_weights', nargs=2, type=float, default=[0.7, 0.3],
                        help="Fusion weights: [prediction error weight, similarity weight]. Must sum to 1.")

    # Debug mode
    parser.add_argument('--video_num_debug', type=int, default=0,
                        help="Number of videos to process during debugging. 0 means all videos.")

    return parser.parse_args()


def run_seg(args):
    # Step 0: Retrieve input paths and parameters
    # The CSV file specified by pred_loss_csv_path should have the following format:
    # Each row corresponds to one video, with the following key columns:
    #   - feature_path: Path to the pre-extracted feature (.npy) file for the video
    #   - gt_path: Path to the ground-truth label file
    #   - start_idx: Start frame index in the video
    #   - end_idx: End frame index in the video
    #   - loss_each: A list or path to a file containing per-frame prediction losses
    pred_loss_csv_path = args.pred_loss_csv_path  # Path to the CSV file containing per-frame prediction loss
    data_root = args.data_root  # Root path of the raw videos or annotation files (optional)
    feature_root = args.feature_root  # Root directory where feature files are stored
    feature_dim = args.feature_dim  # Dimensionality of the features
    save_root = args.save_root  # Path to save the segmentation results

    # Configuration for PS-Seg segmentation
    ps_seg_params = {
        "dcp": args.dcp,  # Change point detection method (e.g., 'knee', 'delta', 'auto')
        "refine": args.refine,  # Refinement method for segmentation (e.g., clustering strategy)
        "k_type": args.k_type,  # Strategy to determine the number of clusters
        "w_size": args.w_size,  # Smoothing window size
        "fuse_weights": args.fuse_weights  # Fusion weights: [prediction error weight, similarity weight]
    }

    # Initialize the PS-Seg
    psseg = PSSeg(
        pred_loss_csv_path,
        data_root,
        feature_root,
        feature_dim,
        save_root,
        ps_seg_params,
        args
    )

    # If debug mode is enabled, only process the first few videos
    video_num_debug = args.video_num_debug if args.video_num_debug > 0 else None

    # Run segmentation and save results
    psseg.segment(video_num_debug)


if __name__ == '__main__':
    seed = 42
    seed_everything(seed)

    args = parse_args()

    pprint(vars(args))

    run_seg(args)
