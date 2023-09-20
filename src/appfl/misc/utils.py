import torch
import os
from omegaconf import DictConfig
import logging
import random
import numpy as np
import copy

def validation(self, dataloader, metric):
    if self.loss_fn is None or dataloader is None:
        return 0.0, 0.0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    validation_model = copy.deepcopy(self.model)
    validation_model.to(device)
    validation_model.eval()

    loss, tmpcnt = 0, 0
    with torch.no_grad():
        for img, target in dataloader:
            tmpcnt += 1
            img = img.to(device)
            target = target.to(device)
            output = validation_model(img)
            loss += self.loss_fn(output, target).item()
    loss = loss / tmpcnt
    accuracy = _evaluate_model_on_tests(validation_model, dataloader, metric)
    return loss, accuracy

def _evaluate_model_on_tests(model, test_dataloader, metric):
    if metric is None:
        metric = _default_metric
    model.eval()
    with torch.no_grad():
        test_dataloader_iterator = iter(test_dataloader)
        y_pred_final = []
        y_true_final = []
        for (X, y) in test_dataloader_iterator:
            if torch.cuda.is_available():
                X = X.cuda()
                y = y.cuda()
            y_pred = model(X).detach().cpu()
            y = y.detach().cpu()
            y_pred_final.append(y_pred.numpy())
            y_true_final.append(y.numpy())

        y_true_final = np.concatenate(y_true_final)
        y_pred_final = np.concatenate(y_pred_final)
        accuracy = float(metric(y_true_final, y_pred_final))
    return accuracy

def _default_metric(y_true, y_pred):
    if len(y_pred.shape) == 1:
        y_pred = np.round(y_pred)
    else:
        y_pred = y_pred.argmax(axis=1, keepdims=False)
    return 100*np.sum(y_pred==y_true)/y_pred.shape[0]

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
    

def load_model_state(cfg: DictConfig, model, client_id = None):
    
    # This function allows to use partial model weights into a model.
    # Useful since server model will only have shared layer weights when personalization is enabled.
    if client_id == None:
        file = cfg.load_model_dirname + "/%s%s" % (cfg.load_model_filename, ".pt")
    else:
        file = cfg.load_model_dirname + "/client_%d"%client_id + "/%s%s" % (cfg.load_model_filename, ".pt")

    model.load_state_dict(torch.load(file),strict=False)
        
    return model
    

def save_model_state_iteration(t, model, cfg: DictConfig, client_id = None):
    
    # This function saves the model weights (instead of the entire model).
    # If personalization is enabled, only the shared layer weights will be saved for the server.
    dir = cfg.save_model_dirname
    if os.path.isdir(dir) == False:
        os.mkdir(dir)
    if client_id != None:
        if os.path.isdir(dir+'/client_%d'%client_id) == False:
            try:
                os.mkdir(dir+'/client_%d'%client_id)
            except:
                pass

    file_ext = ".pt"
    if client_id == None:
        file = dir + "/%s_Round_%s%s" % (cfg.save_model_filename, t, file_ext)
    else:
        file = dir + "/client_%d"%client_id + "/%s_Round_%s%s" % (cfg.save_model_filename, t, file_ext)
    uniq = 1
    while os.path.exists(file):
        file = dir + "/%s_Round_%s_%d%s" % (cfg.save_model_filename, t, uniq, file_ext)
        uniq += 1
        
    state_dict = copy.deepcopy(model.state_dict())
    if client_id == None:
        if cfg.personalization == True:
            keys = [key for key,_ in enumerate(model.named_parameters())]
            for key in keys:
                if key in cfg.p_layers:
                    _ = state_dict.pop(key)

    torch.save(state_dict, file)
 

def set_seed(seed=233):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
