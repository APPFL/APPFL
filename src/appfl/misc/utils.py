import torch
import os
from omegaconf import DictConfig
import logging
import random
import numpy as np


def validation(self, dataloader):

    if self.loss_fn is None or dataloader is None:
        return 0.0, 0.0

    self.model.to(self.device)
    self.model.eval()

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
            output = self.model(img)
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
        os.mkdir(dir)
    output_filename = cfg.output_filename + "_server"

    file_ext = ".txt"
    filename = dir + "/%s%s" % (output_filename, file_ext)
    uniq = 1
    while os.path.exists(filename):
        filename = dir + "/%s_%d%s" % (output_filename, uniq, file_ext)
        uniq += 1

    logger.setLevel(logging.INFO)
    # Create handlers
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(filename)
    c_handler.setLevel(logging.INFO)
    f_handler.setLevel(logging.INFO)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger


def client_log(dir, output_filename):

    if os.path.isdir(dir) == False:
        os.mkdir(dir)

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
