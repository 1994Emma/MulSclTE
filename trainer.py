import abc
import math
import time

from os import path as osp
import pandas as pd
from scipy import spatial
from tqdm import tqdm
import torch


from dataset import get_dataloader
from logger import Logger
from myutil import Averager, prepare_optimizer, prepare_lr_scheduler, resume_training, \
    init_with_pretrained_model, Timer, _utils_basic_logger, warmup, get_model
from transformer import ClipLevelContrastiveLossModule


class BaseTrainer(object, metaclass=abc.ABCMeta):
    def __init__(self, args):
        self.args = args
        self.logger = Logger(args, osp.join(args.save_path))
        self.train_step = 0
        self.train_epoch = 0

        self.max_steps = None
        self.steps_per_epoch = None

        # data_timer, foward_timer, backward_timer, optim_timer
        self.dt, self.ft = Averager(), Averager()
        self.bt, self.ot = Averager(), Averager()
        self.timer = Timer()

        # train statistics
        self.trlog = {}

    @abc.abstractmethod
    def train(self):
        pass

    @abc.abstractmethod
    def evaluate(self):
        pass

    def try_logging(self, tl1, loss_name=None, tg=None):
        args = self.args
        if self.train_step % args.log_interval == 0:
            print('epoch {}/{}, train {:06g}/{:06g} loss={:.4f}, lr={:.4g}'
                  .format(self.train_epoch,
                          self.args.n_epochs,
                          self.train_step-1,
                          self.max_steps,
                          tl1.item(),
                          self.optimizer.param_groups[0]['lr']))

            if loss_name is None:
                self.logger.add_scalar('train_loss', tl1.item(), self.train_step)
            else:
                self.logger.add_scalar('train_{}'.format(loss_name), tl1.item(), self.train_step)

            if tg is not None:
                self.logger.add_scalar('grad_norm', tg.item(), self.train_step)

            self.logger.dump()

    def save_model(self, name, save_tar=True):
        if save_tar:
            if self.lr_scheduler:
                torch.save(
                    dict(params=self.model.state_dict(),
                         epoch=self.train_epoch,
                         train_step=self.train_step,
                         optimizer=self.optimizer.state_dict(),
                         lr_scheduler=self.lr_scheduler.state_dict(),
                         trlog=self.trlog),
                    osp.join(self.args.save_path, name + '.pth.tar')
                )
            else:
                torch.save(
                    dict(params=self.model.state_dict(),
                         epoch=self.train_epoch,
                         train_step=self.train_step,
                         optimizer=self.optimizer.state_dict(),
                         lr_scheduler={},
                         trlog=self.trlog),
                    osp.join(self.args.save_path, name + '.pth.tar')
                )

        # save model
        torch.save(
            dict(params=self.model.state_dict()),
            osp.join(self.args.save_path, name + '.pth')
        )

        _utils_basic_logger.info("Save model epoch={}, train_step={} to {}".format(self.train_epoch, self.train_step,
                                                                                    osp.join(self.args.save_path)))

    def __str__(self):
        return "{}({})".format(
            self.__class__.__name__,
            self.model.__class__.__name__
        )


class Pretrainer(BaseTrainer):
    def __init__(self, args, test_dataloader=None):
        super().__init__(args)

        # Get dataloader
        self.dataloader = get_dataloader(self.args.data_root, self.args.feature_type, self.args.batch_size,
                                         using_clip=self.args.using_clip, shuffle=True, args=self.args)

        # Get model
        self.model = get_model(args)

        # Flags: indicating whether training the model with clip-level contrastive loss
        self.use_cntrst_loss = self.args.use_cntrst
        self.use_cntrst_model = "cntrst" in self.args.model_type

        if args.init_weights is not None:
            self.model = init_with_pretrained_model(self.model, self.args.init_weights, self.args)
            _utils_basic_logger.info(
                "init model from {}".format(self.args.init_weights))

        # Get optimizer & lr_scheduler
        self.optimizer = prepare_optimizer(self.model, args)
        if self.args.lr_scheduler == "None":
            self.lr_scheduler = None
        else:
            self.lr_scheduler = prepare_lr_scheduler(self.optimizer, args)

        self.loss_fn = torch.nn.MSELoss(reduction='mean')
        self.loss_fn_eval = torch.nn.MSELoss(reduction='none')

        if self.use_cntrst_loss:
            self.cntrst_loss_func = ClipLevelContrastiveLossModule(self.args.cntrst_temperature)

        # Resume training
        if self.args.resume:
            self.model, self.optimizer, self.lr_scheduler, self.train_epoch, self.train_step, self.trlog = resume_training(
                self.model, self.optimizer, self.lr_scheduler, self.args)
            _utils_basic_logger.info(
                "resume training from epoch={}, train_step={}".format(self.train_epoch, self.train_step))

        self.trlog['step_loss'] = 0.0
        self.trlog['epoch_loss'] = 0.0

        self.test_dataloader = test_dataloader

    def train(self):
        if self.args.model_type in ["cntrst_bi_encoder", ]:
            self.train_core()
        else:
            raise Exception("Invalid model_type:{}".find(self.args.model_type))

    def evaluate(self, test_dataloader, output_csv=None):
        if self.args.model_type in ["cntrst_bi_encoder", ]:
            self.evaluate_core(test_dataloader, output_csv)
        else:
            raise Exception("Invalid model_type:{}".find(self.args.model_type))

    def train_core(self):
        args = self.args
        self.model.train()

        self.steps_per_epoch = len(self.dataloader)
        self.max_steps = args.n_epochs * len(self.dataloader)

        print("============Start Training============")
        for epoch in range(self.train_epoch, args.n_epochs):
            self.model.train()

            self.train_epoch += 1

            # Initialize loss trackers
            tl1, tl2, tl3, tl4 = Averager(), Averager(), Averager(), Averager()

            start_tm = time.time()
            for batch in self.dataloader:
                self.train_step += 1

                if args.warmup:
                    # warmup optimizer
                    self.optimizer = warmup(self.train_step, self.optimizer, args)

                x_data, _, _, _, _, _, _ = batch
                _utils_basic_logger.debug("x_data.size={}".format(x_data.shape))

                x = x_data.clone().detach().requires_grad_(True)

                # This is ground truth y
                y = x_data.clone().detach().requires_grad_(True)

                batch, seq, dim = y.shape

                _utils_basic_logger.debug("x.size={}, y.size={}".format(x.shape, y.shape))

                data_tm = time.time()
                self.dt.add(data_tm - start_tm)

                # Generate predictions
                f_y_pred, b_y_pred, fb_y_pred, fb_clip_features = self.model(x)

                # Loss1: Compute global-level prediction loss
                y = y.reshape(-1, dim)
                fb_y_pred = fb_y_pred.reshape(-1, dim)
                _utils_basic_logger.debug("fb_y_pred.size={}, b_y.size={} ".format(fb_y_pred.shape, y.shape))
                fb_pred_loss = self.loss_fn(fb_y_pred, y)
                _utils_basic_logger.debug("fb_pred_loss={} ".format(fb_pred_loss))

                pred_loss = fb_pred_loss
                tl2.add(pred_loss.item())

                # Loss2: Compute clip-level contrastive loss
                cntrst_loss = None
                if self.use_cntrst_model and self.use_cntrst_loss:
                    batch_size = fb_clip_features.size(0)
                    fb_cntrst_loss = None
                    for i in range(batch_size):
                        c_fb_loss = self.cntrst_loss_func(fb_clip_features[i, :, :])
                        if fb_cntrst_loss is None:
                            fb_cntrst_loss = c_fb_loss
                        else:
                            fb_cntrst_loss += c_fb_loss

                    _utils_basic_logger.debug("fb_cntrst_loss={}".format(fb_cntrst_loss/batch_size))
                    cntrst_loss = fb_cntrst_loss / batch_size

                # Compute total loss
                if cntrst_loss is None:
                    loss = pred_loss
                    print_str = ", pred_loss={}".format(pred_loss)
                else:
                    loss = pred_loss * self.args.pred_loss_weight + cntrst_loss * self.args.cntrst_loss_weight

                    print_str = ", pred_loss={}, cntrst_loss={}".format(pred_loss, cntrst_loss)
                    tl3.add(cntrst_loss.item())

                print_str = "Epoch={}, step={}, loss={}".format(self.train_epoch, self.train_step, loss) + print_str
                _utils_basic_logger.info(print_str)
                tl1.add(loss.item())

                forward_tm = time.time()
                self.ft.add(forward_tm - data_tm)

                # backward
                self.optimizer.zero_grad()
                loss.backward()
                backward_tm = time.time()
                self.bt.add(backward_tm - forward_tm)

                self.optimizer.step()
                optimizer_tm = time.time()
                self.ot.add(optimizer_tm - backward_tm)

                # refresh start_tm
                start_tm = time.time()

                self.try_logging(tl1, "loss")

                if cntrst_loss:
                    self.try_logging(tl2, "pred_loss")
                    self.try_logging(tl3, "cntrst_loss")

            if (args.warmup and self.train_step >= math.ceil(
                    1.0 * args.warmup_max_steps / self.steps_per_epoch) * self.steps_per_epoch) or (not args.warmup):
                # only running lr_scheduler after warmup
                if self.lr_scheduler:
                    _utils_basic_logger.info(
                        "epoch={}, step={}, running self.lr_scheduler.step()".format(self.train_epoch,
                                                                                     self.train_step))
                    self.lr_scheduler.step()

            # only save model after warmup
            self.save_model('epoch-{}'.format(self.train_epoch), save_tar=False)
            self.save_model('epoch-last'.format(self.train_epoch))

            if self.test_dataloader is not None:
                output_csv = 'predict_loss_each_epoch_{}.csv'.format(self.train_epoch)
                self.evaluate_core(self.test_dataloader, output_csv)

            print('ETA:{}/{}'.format(self.timer.measure(), self.timer.measure(self.train_epoch / args.n_epochs)))

    def get_similarity_of_adjacent_frames(self, x_data):
        similarities = []
        for i in range(x_data.shape[1] - 1):
            cos_sim = 1 - spatial.distance.cosine(x_data[:, i,:], x_data[:, i + 1, :])
            similarities.append(cos_sim)
        return similarities

    def evaluate_core(self, test_dataloader, output_csv=None):
        # restore model args
        args = self.args

        # evaluation mode
        print("============Start Testing============")
        self.model.eval()
        df_output = []
        with torch.no_grad():
            for i, batch in enumerate(tqdm(test_dataloader), 1):
                x_data, y_action_label, y_data_eval, c_feature_path, c_gt_path, start_idx, end_idx = batch
                _utils_basic_logger.debug("x_data.size={}".format(x_data.shape))

                # input
                x = x_data.clone().detach()
                # this is ground-truth y
                y = x_data.clone().detach()

                _utils_basic_logger.debug("x.size={}, y.size={}".format(x.shape, y.shape))

                # forward
                f_y_pred, b_y_pred, fb_y_pred = self.model.forward_transformer(x)

                b, t, dim = x.shape
                y = y.reshape(-1, dim)

                # compute frame-level prediction errors
                fb_y_pred = fb_y_pred.reshape(-1, dim)
                _utils_basic_logger.debug("fb_y_pred.size={}, y.size={} ".format(fb_y_pred.shape, y.shape))
                fb_loss = self.loss_fn_eval(fb_y_pred, y)
                fb_loss_each = torch.mean(fb_loss, dim=1)
                _utils_basic_logger.debug("fb_loss_each.reshape.size={}".format(fb_loss_each.shape))
                fb_loss_each = fb_loss_each.tolist()[1:]

                # compute adjacent frame similarity
                fb_y_pred = fb_y_pred.reshape(b, t, dim)
                fb_sim = self.get_similarity_of_adjacent_frames(fb_y_pred.cpu())

                df_output.append([c_feature_path, c_gt_path, start_idx.item(), end_idx.item(),
                                  fb_loss_each, fb_sim])

        df = pd.DataFrame(df_output, columns=["feature_path", "gt_path", "start_idx", "end_idx", "loss_each", "pred_fb_sim"])

        if output_csv is None:
            output_csv = 'predict_loss_each.csv'

        df.to_csv(osp.join(self.args.save_path, output_csv), index=False)
        _utils_basic_logger.debug("Done evaluate")