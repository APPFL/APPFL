from ast import With
from asyncio.log import logger
from datetime import datetime
from logging import handlers
from pickle import NONE
import torch
import os
import os.path as osp
from omegaconf import DictConfig
import logging
import random
import numpy as np
import copy
import string
import torch
import pickle as pkl

def get_executable_func(func_cfg):
    if func_cfg.module != "":
        import importlib
        mdl = importlib.import_module(func_cfg.module)
        return getattr(mdl, func_cfg.call)
    elif func_cfg.source != "":
        exec(func_cfg.source, globals())
        return eval(func_cfg.call)

def validation(self, dataloader):

    if self.loss_fn is None or dataloader is None:
        return 0.0, 0.0

    eval_model = copy.deepcopy(self.model)
    eval_model.to(self.device)
    eval_model.eval()

    loss = 0
    correct = 0
    tmpcnt = 0
    tmptotal = 0
    with torch.no_grad():
        for img, target in dataloader:
            tmpcnt += 1
            tmptotal += len(target)
            img = img.to(self.device)
            target = target.to(self.device)
            output = eval_model(img)
            loss += self.loss_fn(output, target).item()

            if output.shape[1] == 1:
                pred = torch.round(output)
            else:
                pred = output.argmax(dim=1, keepdim=True)

            correct += pred.eq(target.view_as(pred)).sum().item()

    # FIXME: do we need to sent the model to cpu again?
    # self.model.to("cpu")

    loss = loss / tmpcnt
    accuracy = 100.0 * correct / tmptotal

    return loss, accuracy


def create_custom_logger(logger, cfg: DictConfig):
    dir = cfg.output_dirname
    if os.path.isdir(dir) == False:
        os.makedirs(dir, exist_ok = True)
    output_filename = cfg.output_filename + "_server"
    
    # TODO: use timestamp instead
    file_ext = ".txt"
    filename = dir + "/%s%s" % (output_filename, file_ext)
    uniq = 1
    while os.path.exists(filename):
        filename = dir + "/%s_%d%s" % (output_filename, uniq, file_ext)
        uniq += 1

    fmt = logging.Formatter(
        '%(asctime)s %(levelname)-8s %(message)s'
    )

    logger.setLevel(logging.INFO)
    # Create handlers
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(filename)
    c_handler.setLevel(logging.INFO)
    f_handler.setLevel(logging.INFO)
    c_handler.setFormatter(fmt)
    f_handler.setFormatter(fmt)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)
    return logger
            
def client_log(dir, output_filename):
    if os.path.isdir(dir) == False:
        os.makedirs(dir, exist_ok = True)
    file_ext = ".txt"
    filename = dir + "/%s%s" % (output_filename, file_ext)
    uniq = 1
    while os.path.exists(filename):
        filename = dir + "/%s_%d%s" % (output_filename, uniq, file_ext)
        uniq += 1

    outfile = open(filename, "a")

    return outfile


def load_model(cfg: DictConfig):
    file = cfg.load_model_dirname + "/%s%s" % (cfg.load_model_filename, ".pt")
    model = torch.load(file)
    model.eval()
    return model


def save_model_iteration(t, model, cfg: DictConfig):
    dir = cfg.save_model_dirname
    if os.path.isdir(dir) == False:
        os.mkdir(dir)

    file_ext = ".pt"
    file = dir + "/%s_Round_%s%s" % (cfg.save_model_filename, t, file_ext)
    uniq = 1
    while os.path.exists(file):
        file = dir + "/%s_Round_%s_%d%s" % (cfg.save_model_filename, t, uniq, file_ext)
        uniq += 1

    torch.save(model, file)
 

def set_seed(seed=233):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

TORCH_EXT = ['.pt', '.pth']
PICKLE_EXT= ['.pkl']

def load_data_from_file(file_path: str, to_device=None):
    """Read data from file using the corresponding readers"""
    # Load files to memory
    file_ext = osp.splitext(osp.basename(file_path))[-1]
    if  file_ext in TORCH_EXT:
        results = torch.load(file_path, map_location=to_device)
    elif file_ext in PICKLE_EXT:
        with open(file_path, "rb") as fi:
            results = pkl.load(fi)
    else:
        raise RuntimeError("File extension %s is not supported" % file_ext)
    return results

def dump_data_to_file(obj, file_path: str):
    """Write data to file using the corresponding readers"""
    file_ext = osp.splitext(osp.basename(file_path))[-1]
    if file_ext in TORCH_EXT:
        torch.save(obj, file_path)
    elif file_ext in PICKLE_EXT:
        with open(file_path, "wb") as fo:
            pkl.dump(obj, fo)
    else:
        raise RuntimeError("File extension %s is not supported" % file_ext)
    return True

from torch.utils.data import DataLoader
def get_dataloader(cfg, dataset, mode):
    """ Create a data loader object from the dataset and config file"""
    if dataset is None:
        return None
    if len(dataset) == 0:
        return None
    assert mode in ['train', 'val', 'test']
    if mode == 'train':
        ## Configure training at client
        batch_size = cfg.train_data_batch_size
        shuffle    = cfg.train_data_shuffle
    else:
        batch_size = cfg.test_data_batch_size
        shuffle    = cfg.test_data_shuffle

    return DataLoader(
            dataset,
            batch_size  = batch_size,
            num_workers = cfg.num_workers,
            shuffle     = shuffle,
            pin_memory  = True
        )

def load_source_file(file_path):
    with open(file_path) as fi:
        source = fi.read()
    return source

def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))