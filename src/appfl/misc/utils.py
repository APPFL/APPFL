import torch
import os
from omegaconf import DictConfig
import logging

def validation(self, dataloader):

    if dataloader is not None:
        self.loss_fn = eval(self.loss_type)        
    else:
        self.loss_fn = None

    if self.loss_fn is None or dataloader is None:
        return 0.0, 0.0

    self.model.to(self.device)
    self.model.eval()
    test_loss = 0
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
            test_loss += self.loss_fn(output, target).item()                
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    # FIXME: do we need to sent the model to cpu again?
    # self.model.to("cpu")

    test_loss = test_loss / tmpcnt
    accuracy = 100.0 * correct / tmptotal

    return test_loss, accuracy

def create_custom_logger(logger, cfg: DictConfig):

    dir = cfg.output_dirname
    if os.path.isdir(dir) == False:
        os.mkdir(dir)
    output_filename = cfg.output_filename 

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

    # # Create formatters and add it to handlers
    # c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    # f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # c_handler.setFormatter(c_format)
    # f_handler.setFormatter(f_format)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)
    
    return logger

def log_title():
    title = "%12s %12s %12s %12s %12s %12s %12s %12s %12s %12s %12s %12s" % (
        "Iter",
        "Local[s]",
        "Global[s]",
        "Valid[s]",
        "Iter[s]",
        "Elapsed[s]",
        "TestAvgLoss",
        "TestAccuracy",
        "Prim_res",
        "Dual_res",
        "Penal_min",
        "Penal_max"
    )
    return title
    
def log_iteration(
    t,
    LocalUpdate_time,
    GlobalUpdate_time,
    Validation_time,
    PerIter_time,
    Elapsed_time,
    test_loss,
    accuracy,
    prim_res,
    dual_res,
    rho_min,
    rho_max,):

    log_iter = "%12d %12.2f %12.2f %12.2f %12.2f %12.2f %12.6f %12.2f %12.4e %12.4e %12.2f %12.2f" % (
        t + 1,
        LocalUpdate_time,
        GlobalUpdate_time,
        Validation_time,
        PerIter_time,
        Elapsed_time,
        test_loss,
        accuracy,
        prim_res,
        dual_res,
        rho_min,
        rho_max,
    )
    return log_iter

def log_summary(logger, cfg: DictConfig,comm_size,
    DataSet_name,
    num_clients,
    Elapsed_time,
    BestAccuracy,):

    logger.info("Device=%s" % (cfg.device))
    logger.info("#Processors=%s" % (comm_size))
    logger.info("Dataset=%s" % (DataSet_name))
    logger.info("#Clients=%s" % (num_clients))
    logger.info("Algorithm=%s" % (cfg.fed.type))
    logger.info("Comm_Rounds=%s" % (cfg.num_epochs))
    logger.info("Local_Epochs=%s" % (cfg.fed.args.num_local_epochs))
    logger.info("DP_Eps=%s" % (cfg.fed.args.epsilon))    
    logger.info("Clipping=%s" % (cfg.fed.args.clip_value))    
    logger.info("Elapsed_time=%s" % (round(Elapsed_time, 2)))
    logger.info("BestAccuracy=%s" % (round(BestAccuracy, 2)))  

    
def load_model(cfg: DictConfig):
    file = cfg.load_model_dirname + "/%s%s" %(cfg.load_model_filename, ".pt")    
    model = torch.load(file)    
    model.eval()
    return model    

def save_model_iteration(t, model, cfg: DictConfig):
    dir = cfg.save_model_dirname
    if os.path.isdir(dir) == False:
        os.mkdir(dir)
    model_name=cfg.save_model_filename
    
    file_ext = ".pt"
    file = dir + "/%s_Round_%s%s" % (model_name, t, file_ext)
    uniq = 1
    while os.path.exists(file):
        file = dir + "/%s_%d%s" % (model_name, uniq, file_ext)
        uniq += 1
     
    torch.save(model, file)    
 