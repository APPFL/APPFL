import os
import sys
import copy
import torch
import random
import string
import logging
import pathlib
import numpy as np
import pickle as pkl
import os.path as osp
import importlib.util
from omegaconf import DictConfig
from .deprecation import deprecated

def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

def create_instance_from_file(file_path, class_name, *args, **kwargs):
    """
    Creates an instance of a class from a given file path.

    :param file_path: The file path where the class is defined.
    :param class_name: The name of the class to be instantiated.
    :param args: Positional arguments to be passed to the class constructor.
    :param kwargs: Keyword arguments to be passed to the class constructor.
    :return: An instance of the specified class, or None if creation fails.
    """
    # Normalize the file path
    file_path = os.path.abspath(file_path)

    # Check if the file exists
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Extract module name and directory from the file path
    module_dir, module_file = os.path.split(file_path)
    module_name, _ = os.path.splitext(module_file)

    # Add module directory to sys.path
    if module_dir not in sys.path:
        sys.path.append(module_dir)

    # Load the module
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Get the class and create an instance
    cls = getattr(module, class_name)
    instance = cls(*args, **kwargs)

    return instance

def get_function_from_file(file_path, function_name):
    """
    Gets a function from a given file path.

    :param file_path: The file path where the function is defined.
    :param function_name: The name of the function to be retrieved.
    :return: The function object, or None if retrieval fails.
    """
    try:
        # Normalize the file path
        file_path = os.path.abspath(file_path)

        # Check if the file exists
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Extract module name and directory from the file path
        module_dir, module_file = os.path.split(file_path)
        module_name, _ = os.path.splitext(module_file)

        # Add module directory to sys.path
        if module_dir not in sys.path:
            sys.path.append(module_dir)

        # Load the module
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Get the function
        function = getattr(module, function_name)

        return function

    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
def run_function_from_file(file_path, function_name, *args, **kwargs):
    """
    Runs a function from a given file path.

    :param file_path: The file path where the function is defined.
    :param function_name: The name of the function to be executed.
    :param args: Positional arguments to be passed to the function.
    :param kwargs: Keyword arguments to be passed to the function.
    :return: The result of the function execution, or None if execution fails.
    """
    try:
        # Normalize the file path
        file_path = os.path.abspath(file_path)

        # Check if the file exists
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Extract module name and directory from the file path
        module_dir, module_file = os.path.split(file_path)
        module_name, _ = os.path.splitext(module_file)

        # Add module directory to sys.path
        if module_dir not in sys.path:
            sys.path.append(module_dir)

        # Load the module
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Get the function and run it
        function = getattr(module, function_name)
        result = function(*args, **kwargs)

        return result

    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
def create_instance_from_file_source(source, class_name, *args, **kwargs):
    """
    Creates an instance of a class from a given source code.

    :param source: The source code where the class is defined.
    :param class_name: The name of the class to be instantiated.
    :param args: Positional arguments to be passed to the class constructor
    """
    # Create a temporary file to store the source code
    _home = pathlib.Path.home()
    dirname = osp.join(_home, ".appfl", "tmp")
    if not osp.exists(dirname):
        pathlib.Path(dirname).mkdir(parents=True, exist_ok=True)
    file_path = osp.join(dirname, f"{id_generator()}.py")
    with open(file_path, "w") as file:
        file.write(source)

    # Create an instance from the temporary file
    instance = create_instance_from_file(file_path, class_name, *args, **kwargs)

    # Remove the temporary file
    os.remove(file_path)

    return instance

def get_function_from_file_source(source, function_name):
    """
    Gets a function from a given source code.

    :param source: The source code where the function is defined.
    :param function_name: The name of the function to be retrieved.
    :return: The function object, or None if retrieval fails.
    """
    # Create a temporary file to store the source code
    _home = pathlib.Path.home()
    dirname = osp.join(_home, ".appfl", "tmp")
    if not osp.exists(dirname):
        pathlib.Path(dirname).mkdir(parents=True, exist_ok=True)
    file_path = osp.join(dirname, f"{id_generator()}.py")
    with open(file_path, "w") as file:
        file.write(source)

    # Get the function from the temporary file
    function = get_function_from_file(file_path, function_name)

    # Remove the temporary file
    os.remove(file_path)

    return function

def run_function_from_file_source(source, function_name, *args, **kwargs):
    """
    Runs a function from a given source code.

    :param source: The source code where the function is defined.
    :param function_name: The name of the function to be executed.
    :param args: Positional arguments to be passed to the function.
    :param kwargs: Keyword arguments to be passed to the function.
    :return: The result of the function execution, or None if execution fails.
    """
    function = get_function_from_file_source(source, function_name)
    if function is None:
        return None
    result = function(*args, **kwargs)
    return result

def get_unique_filename(
    dirname: str,
    filename: str,
):
    """
    Create the directory (if needed) and get a unique filename by appending a number to the filename.
    :param dirname: The directory where the file is located.
    :param filename: The original filename.
    :return dirname, unique_filename: The directory and the unique filename.
    """
    if not osp.exists(dirname):
        pathlib.Path(dirname).mkdir(parents=True, exist_ok=True)
    unique = 1
    unique_filename = filename
    filename_base, ext = osp.splitext(filename)
    while pathlib.Path(osp.join(dirname, unique_filename)).exists():
        unique_filename = f"{filename_base}_{unique}{ext}"
        unique += 1
    return dirname, unique_filename

def load_data_from_file(
    file_path: str, 
    to_device=None
):
    """
    Read data from file using the corresponding readers.
    For uncompressed model weights of PyTorch models, the weights are stored in a dictionary in the `pt` or `pth` file.
    For compressed model weights of PyTorch models, the weights are stored as bytes in the `pkl` file.
    """
    TORCH_EXT = ['.pt', '.pth']
    PICKLE_EXT= ['.pkl']
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
    """
    Write data to file using the corresponding readers
    """
    TORCH_EXT = ['.pt', '.pth']
    PICKLE_EXT= ['.pkl']
    file_ext = osp.splitext(osp.basename(file_path))[-1]
    if file_ext in TORCH_EXT:
        torch.save(obj, file_path)
    elif file_ext in PICKLE_EXT:
        with open(file_path, "wb") as fo:
            pkl.dump(obj, fo)
    else:
        raise RuntimeError("File extension %s is not supported" % file_ext)
    return True

def save_partial_model_iteration(t, model, cfg: DictConfig, client_id = None):
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

def model_parameters_clip_factor(model, pre_update_params, C, norm_type=1):
    """
    Compute the norm of all the parameters of a model and return that C / norm
    if norm is below the threshold C, otherwise return 1.
    :model: The PyTorch model
    :C: The threshold value
    :param norm_type: The type of norm to compute (default is 2, which is the L2 norm)
    """
    total_norm = 0.0
    for param_old, param_new in zip(pre_update_params,list(model.parameters())):
        param_norm = torch.norm(param_old-param_new,p=norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return ( C / total_norm ) if total_norm > C else 1

def scale_update(model, pre_update_params, scale = 1.0):
    """
    Scale update of <<updated>> model.
    :model: <<updated>> model
    :pre_update_params: list of <<pre-update>> model parameters
    :scale: Scaling factor
    """
    for param_old, param_new in zip(pre_update_params,list(model.parameters())):
        param_new = param_old + scale * (param_new - param_old)

@deprecated(silent=True)
def set_seed(seed=233):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@deprecated(silent=True)
def load_source_file(file_path):
    with open(file_path) as fi:
        source = fi.read()
    return source

@deprecated(silent=True)
def compute_gradient(original_model, trained_model):
    """Compute the difference (i.e. gradient) between the original model and the trained model"""
    list_named_parameters = []
    for name, _ in trained_model.named_parameters():
        list_named_parameters.append(name)
    local_gradient = {}
    trained_model_state_dict = trained_model.state_dict()
    for name in trained_model_state_dict:
        if name in list_named_parameters:
            local_gradient[name] = original_model[name] - trained_model_state_dict[name]
        else:
            local_gradient[name] = trained_model_state_dict[name]
    return local_gradient    

@deprecated(silent=True)
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

@deprecated(silent=True)
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

@deprecated(silent=True)
def _default_metric(y_true, y_pred):
    if len(y_pred.shape) == 1:
        y_pred = np.round(y_pred)
    else:
        y_pred = y_pred.argmax(axis=1)
    return 100*np.sum(y_pred==y_true)/y_pred.shape[0]

@deprecated(silent=True)
def create_custom_logger(logger, cfg: DictConfig):

    dir = cfg.output_dirname
    if os.path.isdir(dir) == False:
        os.makedirs(dir, exist_ok=True)
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

@deprecated(silent=True)
def client_log(dir, output_filename):

    if os.path.isdir(dir) == False:
        os.makedirs(dir, exist_ok=True)

    file_ext = ".txt"
    filename = dir + "/%s%s" % (output_filename, file_ext)
    uniq = 1
    while os.path.exists(filename):
        filename = dir + "/%s_%d%s" % (output_filename, uniq, file_ext)
        uniq += 1

    outfile = open(filename, "a")

    return outfile

@deprecated(silent=True)
def load_model(cfg: DictConfig):
    
    file = cfg.load_model_dirname + "/%s%s" % (cfg.load_model_filename, ".pt")
    model = torch.load(file)
    model.eval()
    return model

@deprecated(silent=True)
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

@deprecated(silent=True)
def load_model_state(cfg: DictConfig, model, client_id = None):
    
    # This function allows to use partial model weights into a model.
    # Useful since server model will only have shared layer weights when personalization is enabled.
    if client_id == None:
        file = cfg.load_model_dirname + "/%s%s" % (cfg.load_model_filename, ".pt")
    else:
        file = cfg.load_model_dirname + "/client_%d"%client_id + "/%s%s" % (cfg.load_model_filename, ".pt")

    model.load_state_dict(torch.load(file),strict=False)
        
    return model