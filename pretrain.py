"""
Training and Evaluating the Bi-Encoder Model
This script allows for training, testing, and feature encoding of the Bi-Encoder model.
"""
import argparse
import torch
import os

from tester import Tester

os.environ['NUMEXPR_MAX_THREADS'] = '16'

from dataset import get_dataloader
from myutil import seed_everything, pprint, set_gpu, make_dirs
from trainer import Pretrainer


def parse_args():
    parser = argparse.ArgumentParser()
    # Operation mode
    parser.add_argument('--operate', type=str, default="train", help="Specify operation mode",
                        choices=[
                            "train",  # Train the model using the specified dataset and configuration.
                            "test-encode",  # Generate features for the entire video at once.
                        ])

    parser.add_argument('--data_root', type=str, help="Path to the root directory of the dataset")
    parser.add_argument('--ds_name', type=str, default="Breakfast", help="Name of the dataset",
                        choices=["Breakfast", "50Salads", "YTI", "epich"])
    parser.add_argument('--save_path', type=str, help="Path to save the output")
    parser.add_argument('--feature_type', type=str, default="i3d", help="Type of input feature", choices=["i3d", ])
    parser.add_argument('--feature_save_root', type=str, help="Path to save the output features")

    # related to input data
    parser.add_argument('--debug_phase', action='store_true',
                        help="Run preliminary experiments on a small subset of the Breakfast dataset")

    parser.add_argument('--max_seq_len', type=int, default=10000, help="Maximum length of input sequences")
    # 2048 for salads and breakfast; 3000 for YTI dataset
    parser.add_argument('--input_dims', type=int, default=2048, help="Dimension of input feature")
    parser.add_argument('--using_clip', action='store_true', help="Use clip generation during input")
    parser.add_argument('--clip_window_size', type=int, default=500, help="Sliding window size for creating clips")
    parser.add_argument('--clip_window_step', type=int, default=100, help="Step size for creating clips")
    parser.add_argument('--zero_padding', action='store_true', help="Use zero padding for input sequences")
    parser.add_argument('--clip_split_method', type=str, default="train_split", help="Method to split input features, "
                                                                                     "valid if using_clip is True",
                        choices=["train_split", ])

    # Bi-Encoder related parameters
    parser.add_argument('--num_of_layers', type=int, default=12, help="Number of attention layers in the transformer")
    parser.add_argument('--heads', type=int, default=4, help="Number of attention heads in transformer layer")
    parser.add_argument('--forward_expansion', type=int, default=4, help="Forward expansion factor in transformer")
    parser.add_argument('--dropout1', type=float, default=0.1, help="Dropout rate in transformer layers")
    parser.add_argument('--layer_norm_eps1', type=float, default=1e-5, help="Epsilon for layer normalization")

    # Training parameters
    parser.add_argument('--n_epochs', type=int, default=1, help="Number of training epochs.")
    parser.add_argument('--batch_size', type=int, default=8,
                        help="Batch size for training: must be 1 for original video; can be any values when "
                             "using_clip is True")

    # Optimizer, scheduler, and warmup parameters
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--betas', type=float, default=(0.9, 0.95), nargs='+', help="Betas for AdamW optimizer")
    parser.add_argument('--weight_decay1', type=float, default=0.1, help="Weight decay for AdamW optimizer")
    parser.add_argument('--lr_mul', type=float, default=10, help="Multiplier for learning rate")
    parser.add_argument('--lr_scheduler', type=str, default='None', choices=['None', 'multistep', 'step', 'cosine'])
    parser.add_argument('--step_size', type=str, default='5')
    # --gamma is used for updating lr, lr=lr * gamma
    parser.add_argument('--gamma', type=float, default=0.2)

    parser.add_argument('--warmup', action='store_true', help="Enable warmup")
    parser.add_argument('--warmup_max_steps', type=int, default=60, help="Number of warmup steps")

    # Gpu parameters
    parser.add_argument('--gpu', default='0', help="GPU device index to use")

    # Logging parameters
    parser.add_argument('--log_interval', default=10, type=int, help="Interval for logging training progress")

    # Model checkpoint and initialization
    parser.add_argument('--resume', action='store_true', help="Resume training from the latest checkpoint")
    parser.add_argument('--init_weights', type=str, default=None, help='Path to initialization weights')

    # Contrastive training parameters
    parser.add_argument('--model_type', type=str, default="cntrst_bi_encoder",  choices=["cntrst_bi_encoder",])

    # Used only when model_type == "cntrst_bi_encoder"
    parser.add_argument('--use_cntrst', action='store_true', help="Enable clip-level contrastive loss")
    parser.add_argument('--hidden_dims', type=int, default=512)
    parser.add_argument('--cntrst_temperature', type=float, default=1.0)
    parser.add_argument("--cntrst_loss_weight", type=float, default=1.0, help="Weight of contrastive loss")
    parser.add_argument("--pred_loss_weight", type=float, default=1.0, help="Weight of prediction loss")
    parser.add_argument('--cntrst_clip_width', type=int, default=5, help="Width of contrastive clips")

    opt = parser.parse_args()

    return opt


if __name__ == '__main__':

    seed = 42
    seed_everything(seed)

    args = parse_args()
    pprint(vars(args))

    set_gpu(args.gpu)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("current device: {}".format(args.device))

    make_dirs(args.save_path)

    if args.operate == "train":
        test_dataloader = get_dataloader(args.data_root, args.feature_type, batch_size=1, using_clip=False,
                                         shuffle=False, args=args, clip_split_method="train_split")
        trainer = Pretrainer(args, test_dataloader)
        trainer.train()
        trainer.evaluate(test_dataloader)
    elif args.operate == "test-encode":
        print("Saving features into: {}".format(args.feature_save_root))
        test_dataloader = get_dataloader(args.data_root, args.feature_type, batch_size=1, using_clip=False,
                                         shuffle=False, args=args)
        tester = Tester(args)
        tester.encode_features(test_dataloader, args.feature_save_root, True)