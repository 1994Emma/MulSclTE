import random
import time
from functools import wraps

import numpy as np
import os
from os import path as osp
import pprint

import torch
from torch import optim
from torch.nn import init

from logger import get_basic_logger
from transformer import BidirectionalTemporalEncoder, Projector_MLP

_utils_pp = pprint.PrettyPrinter()
_utils_basic_logger = get_basic_logger()


#######################################
# Model, Training, and Prediction Utilities
#######################################
def get_parameter_number(net):
    """
    Calculate the total and trainable number of parameters in the model.
    :param net: the model object
    :return: Dictionary with total and trainable parameter counts.
    """
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def prepare_optimizer(model, args):
    """
    Prepare the optimizer with weight decay configuration based on the model's parameters.

    This long function is unfortunately doing something very simple and is being very defensive:
       We are separating out all parameters of the model into two buckets: those that will experience
       weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
       We are then returning the PyTorch optimizer object.

    :param model: The model
    :param args: Arguments including learning rate, weight decay, etc.
    :return: Configured optimizer object.
    """

    # separate out all parameters to those that will and won't experience regularizing weight decay
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear, Projector_MLP)
    blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding, torch.nn.Parameter)
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name
            if (pn.endswith('bias')) or ("bias" in pn):
                # all biases will not be decayed
                no_decay.add(fpn)
            elif (pn.endswith('weight') or ("weight" in pn)) and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif (pn.endswith('weight') or ("weight" in pn)) and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)
            elif "positional_embeddings" in fpn:
                no_decay.add(fpn)

    # Validate parameter separation
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
    assert len(
        param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                % (str(param_dict.keys() - union_params),)

    # Create the optimizer object
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": args.weight_decay1},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=args.lr, betas=args.betas)

    return optimizer


def prepare_lr_scheduler(optimizer, args):
    """
    Prepare the learning rate scheduler based on the specified type.

    :param optimizer:
    :param args:
    :return: Configured learning rate scheduler
    """
    if args.lr_scheduler == 'step':
        lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(args.step_size),
            gamma=args.gamma
        )
    elif args.lr_scheduler == 'multistep':
        lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[int(_) for _ in args.step_size.split(',')],
            gamma=args.gamma,
        )
    elif args.lr_scheduler == 'cosine':
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            args.n_epochs,
            eta_min=0  # a tuning parameter
        )
    elif args.lr_scheduler == "None":
        lr_scheduler = None
    else:
        raise ValueError('No Such Scheduler a')

    return lr_scheduler


def warmup(step, optimizer, args):
    """
    Learning rate warm-up.
    :param step: Current training step
    :param optimizer:
    :param args: Arguments including args.warmup_max_steps represent how many steps to be warmup and learning rate
    :return: Updated optimizer.
    """

    if step < args.warmup_max_steps:
        c_lr = args.lr * (step / float(max(1, args.warmup_max_steps)))
        _utils_basic_logger.debug("warmup step={}/warm_steps={}, lr==={}".format(step, args.warmup_max_steps, c_lr))
        for c_param_group in optimizer.param_groups:
            c_param_group['lr'] = c_lr
    elif step == args.warmup_max_steps:
        _utils_basic_logger.debug("warmup end step={}/warm_steps={}, lr==={}".format(step, args.warmup_max_steps, args.lr))
        for c_param_group in optimizer.param_groups:
            c_param_group['lr'] = args.lr

    return optimizer


def init_with_pretrained_model(model, init_weights_path, args):
    """
    Initialize the model with pre-trained weights.
    :param model: The model to initialize.
    :param init_weights_path: Path to the pre-trained weights.
    :param args: Arguments including device.
    :return: Model initialized with pre-trained weights.
    """
    model_dict = model.state_dict()
    pretrained_dict = torch.load(init_weights_path)['params']
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}

    print(pretrained_dict.keys())
    print(model_dict.keys())

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    model = model.to(args.device)

    return model


def resume_training(model, optimizer, lr_scheduler, args):
    """
    Resume training from the checkpoint.
    :param model: The model to resume.
    :param optimizer: The optimizer to resume.
    :param lr_scheduler: The learning rate scheduler to resume.
    :param args: Arguments including save path.
    :return: Tuple containing the model, optimizer, scheduler, and training state.
    """
    # load checkpoint
    if os.path.exists(osp.join(args.save_path, 'epoch-last.pth.tar')):
        state = torch.load(osp.join(args.save_path, 'epoch-last.pth.tar'))
    else:
        state = torch.load(osp.join(args.save_path, 'max_acc.pth.tar'))

    # resume model
    model.load_state_dict(state['params'])

    train_epoch = state['epoch']
    train_step = state['train_step']
    trlog = state['trlog']

    # resume optimizer and lr_scheduler
    optimizer.load_state_dict(state['optimizer'])

    if lr_scheduler:
        lr_scheduler.load_state_dict(state['lr_scheduler'])

    return model, optimizer, lr_scheduler, train_epoch, train_step, trlog


def get_model(args):
    """
    Get the model based on the specified type.
    :param args: Arguments including model type and parameters.
    :return: Configured model instance.
    """
    if args.model_type == "cntrst_bi_encoder":
        model = BidirectionalTemporalEncoder(args.input_dims, args.hidden_dims, args.dropout1, args.heads,
                                             args.num_of_layers,
                                             args.forward_expansion, args.max_seq_len,
                                             clip_length=args.cntrst_clip_width)
    else:
        raise Exception("Invalid model_type:{}".find(args.model_type))

    return model


#######################################
# Utilities for Value and Time Tracking
#######################################
class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


class Timer:
    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))


def time2human(seconds):
    x = int(seconds)
    if x >= 3600:
        return '{:.1f}h'.format(x / 3600)
    if x >= 60:
        return '{}m'.format(round(x / 60))

    return '{}s'.format(x)


def print_running_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration_time = end_time - start_time
        print("execute time running func_{}: {}s, {}".format(func.__name__, duration_time, time2human(duration_time)))
        return result

    return wrapper


#######################################
# Utilities related to gpu setting
#######################################
def set_gpu(x):
    os.environ['CUDA_VISIBLE_DEVICES'] = x
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    print('using gpu:', x)


#######################################
# Utilities related to initialization
#######################################
def weights_init_normal(m):
    classname = m.__class__.__name__
    # print(classname)
    if (classname.find('Conv') != -1) and (classname != "ConvNet") and (classname !="ConvBlock") and (classname!="ConvNetCBAM"):
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if (classname.find('Conv') != -1) and (classname != "ConvNet") and (classname != "ConvBlock") and (
            classname != "ConvNetCBAM"):
        init.xavier_normal_(m.weight.data, gain=0.02)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data, gain=0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if (classname.find('Conv') != -1) and (classname != "ConvNet") and (classname != "ConvBlock") and (
            classname != "ConvNetCBAM"):
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    print(classname)
    if (classname.find('Conv') != -1) and (classname != "ConvNet") and (classname != "ConvBlock") and (
            classname != "ConvNetCBAM"):
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


#######################################
# Utilities related to seeding
#######################################
def seed_everything(seed):
    """
    setting seed
    :param seed:
    :return:
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


#######################################
# Utilities related to file reader/writer, printer
#######################################
def np_load_file(filepath, dtype="float"):
    """
    load data into nd-array
    :param filepath:
    :param dtype:
    :return:
    """
    suffix = os.path.split(filepath)[-1].split(".")
    data = None
    if (len(suffix) > 1) and (suffix[-1] == "npy"):
        data = np.load(filepath).astype(dtype)
    elif (len(suffix) == 1) or (suffix[-1] == "txt"):
        data = np.loadtxt(filepath, dtype=dtype)
    return data


def np_load_feature(filepath):
    """
    load feature into nd-array
    :param filepath:
    :return:
    """
    return np_load_file(filepath, dtype="float32")


def np_load_gt(filepath):
    """
    load ground truth into nd-array
    :param filepath:
    :return:
    """
    return np_load_file(filepath, dtype="str")


def pprint(x):
    _utils_pp.pprint(x)


def make_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print("create dir: {}".find(path))