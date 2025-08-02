import abc
import torch

import os
from os import path as osp
import pandas as pd
import numpy as np
from tqdm import tqdm

from logger import Logger
from myutil import Averager, init_with_pretrained_model, Timer, _utils_basic_logger, get_model, print_running_time, make_dirs


class BaseTester(object, metaclass=abc.ABCMeta):
    def __init__(self, args):
        self.args = args
        self.logger = Logger(args, osp.join(args.save_path))
        self.train_step = 0
        self.train_epoch = 0

        self.max_steps = None
        self.steps_per_epoch = None

        # data_timer, forward_timer, backward_timer, optim_timer
        self.dt, self.ft = Averager(), Averager()
        self.bt, self.ot = Averager(), Averager()
        self.timer = Timer()

        # train statistics
        self.trlog = {}

    def __str__(self):
        return "{}({})".format(
            self.__class__.__name__,
            self.model.__class__.__name__
        )


class Tester(BaseTester):
    """
    Uses the trained model to predict features for the corresponding input video:
    (1) Calculate the corresponding loss.
    (2) Save the predicted features.
    """
    def __init__(self, args):
        super().__init__(args)

        # Get model
        self.model = get_model(args)

        if args.init_weights is not None:
            self.model = init_with_pretrained_model(self.model, self.args.init_weights, self.args)
            _utils_basic_logger.info(
                "init model from {}".format(self.args.init_weights))

        self.loss_fn_eval = torch.nn.MSELoss(reduction='none')

    @print_running_time
    def encode_features(self, test_dataloader, output_feature_root=None, calc_pred_loss=False):
        """
        Encode features and save them to disk.

        :param test_dataloader: DataLoader for the test data.
        :param output_feature_root: Root directory to save encoded features.
        :param calc_pred_loss: Flag to calculate prediction loss.
        """

        # Evaluation mode
        print("============Start Testing============")
        output_root = os.path.join(output_feature_root, self.args.model_type)

        fb_root = os.path.join(output_root, "fb_merge")
        if not osp.exists(fb_root):
            os.makedirs(fb_root)

        if not osp.exists(fb_root):
            os.makedirs(fb_root)

        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(tqdm(test_dataloader), 1):
                x_data, y_action_label, y_data_eval, c_feature_path, c_gt_path, start_idx, end_idx = batch
                _utils_basic_logger.debug("x_data.size={}".format(x_data.shape))

                c_output_filename = osp.split(c_feature_path[0])[-1]

                x = x_data.clone().detach()
                y = x_data.clone().detach()

                _utils_basic_logger.debug("x.size={}, y.size={}".format(x.shape, y.shape))

                # Forward
                _, _, fb_y_pred = self.model.forward_transformer(x)

                # batch, time_len, dim
                b, t, dim = y.shape
                fb_y_pred = fb_y_pred.reshape(-1, dim).cpu().numpy()

                # Save features
                sub_folder = None
                if "YTI" in self.args.ds_name:
                    sub_folder = osp.split(osp.split(c_feature_path[0])[0])[-1]

                if sub_folder is None:
                    c_output_filename = osp.join(fb_root, c_output_filename)
                else:
                    c_output_root = osp.join(fb_root, sub_folder)
                    if not osp.exists(c_output_root):
                        os.makedirs(c_output_root)
                    c_output_filename = osp.join(c_output_root, c_output_filename)

                np.save(c_output_filename, fb_y_pred.T)

                _utils_basic_logger.debug("fb_y_pred.shape={}".format(fb_y_pred.shape))

        # Calculate prediction errors
        if calc_pred_loss:
            if "YTI" in self.args.ds_name:
                pred_feat_root = osp.split(osp.split(c_feature_path[0])[0])[0]
            else:
                pred_feat_root = osp.split(c_output_filename)[0]
            loss_output_root = osp.join(os.path.split(pred_feat_root)[0], "fb_merge_loss")
            loss_save_path = osp.join(loss_output_root, "prediction_loss.csv")
            self.generate_prediction_loss(test_dataloader, pred_feat_root, loss_save_path)

        _utils_basic_logger.debug("Done evaluate")

    def generate_prediction_loss(self, ori_dataloader, pred_feat_root, loss_output_csv):

        make_dirs(osp.split(loss_output_csv)[0])

        results = []
        for i, batch in enumerate(tqdm(ori_dataloader), 1):
            x_data, _, _, c_feature_path, c_gt_path, start_idx, end_idx = batch
            # batch, time_len, dim
            batch, time_len, dim = x_data.shape
            x_data = x_data.reshape(batch * time_len, dim)

            c_feature_filename = osp.split(c_feature_path[0])[-1]
            _utils_basic_logger.debug("c_feature_filename={}".format(c_feature_filename))

            sub_folder = None
            if "YTI" in self.args.ds_name:
                sub_folder = osp.split(osp.split(c_feature_path[0])[0])[-1]

            if sub_folder is None:
                c_pred_feature_path = osp.join(pred_feat_root, c_feature_filename)
            else:
                c_pred_feature_path = osp.join(pred_feat_root, sub_folder, c_feature_filename)

            _utils_basic_logger.debug("c_pred_feature_path={}".format(c_pred_feature_path))

            if c_pred_feature_path.endswith("npy"):
                c_pred_feat = np.load(c_pred_feature_path)
            elif c_pred_feature_path.endswith("txt"):
                c_pred_feat = np.loadtxt(c_pred_feature_path)

            c_pred_feat = torch.from_numpy(c_pred_feat)

            if c_pred_feat.shape[0] == dim:
                c_pred_feat = c_pred_feat.T

            loss = self.loss_fn_eval(c_pred_feat.cpu(), x_data.cpu())
            loss_each = torch.mean(loss, dim=1)
            loss_each = loss_each.tolist()

            _utils_basic_logger.debug("x_data.shape={}, len(loss)={}".format(x_data.shape, len(loss)))

            results.append([c_feature_path, c_gt_path, 0, len(loss_each), loss_each])

        df = pd.DataFrame(results, columns=["feature_path", "gt_path", "start_idx", "end_idx", "loss_each"])

        # write into file
        df.to_csv(loss_output_csv, index=False)

        _utils_basic_logger.info("Save predict loss into {}".format(loss_output_csv))