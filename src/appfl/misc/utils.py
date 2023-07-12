import torch
import os
from omegaconf import DictConfig
import logging
import random
import numpy as np
import copy


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


def unflatten_model_params(model, flat_params):
    # Convert flat_params to a PyTorch tensor
    flat_params_tensor = torch.from_numpy(flat_params)

    # Get a dictionary of parameter names and shapes from the model's state_dict
    state_dict = model.state_dict()
    param_shapes = {name: param.shape for name, param in state_dict.items()}

    # Initialize a pointer variable to 0
    pointer = 0

    # Create a dictionary to hold the unflattened parameters
    unflattened_params = {}

    # Iterate over the parameters of the model
    for name, param in model.named_parameters():
        # Determine the number of elements in the parameter
        num_elements = param.numel()

        # Slice that number of elements from the flat_params array using the pointer variable
        param_slice = flat_params_tensor[pointer : pointer + num_elements]

        # Reshape the resulting slice to match the shape of the parameter
        param_shape = param_shapes[name]
        param_value = param_slice.view(*param_shape)

        # Update the value of the parameter in the model's state_dict
        state_dict[name] = param_value

        # Add the unflattened parameter to the dictionary
        unflattened_params[name] = param_value

        # Increment the pointer variable by the number of elements used
        pointer += num_elements

    # Load the updated state_dict into the model
    model.load_state_dict(state_dict)

    # Return the dictionary of unflattened parameters
    return unflattened_params


def flatten_model_params(model: torch.nn.Module) -> np.ndarray:
    # Concatenate all of the model's parameters into a 1D tensor
    flat_params = torch.cat([param.view(-1) for param in model.parameters()])
    # Convert the tensor to a numpy array and return it
    return flat_params.detach().cpu().numpy()
