import torch
import os
from omegaconf import DictConfig


def validation(self, dataloader):

    if dataloader is not None:
        self.loss_fn = torch.nn.CrossEntropyLoss()
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
            logits = self.model(img)
            test_loss += self.loss_fn(logits, target).item()
            pred = logits.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    # FIXME: do we need to sent the model to cpu again?
    # self.model.to("cpu")

    test_loss = test_loss / tmpcnt
    accuracy = 100.0 * correct / tmptotal

    return test_loss, accuracy


def print_write_result_title(cfg: DictConfig, DataSet_name: str):
    ## Print and Write Results
    dir = cfg.result_dir
    if os.path.isdir(dir) == False:
        os.mkdir(dir)
    filename = "Result_%s_%s_Batch_%s_PrivEps_%s" % (DataSet_name, cfg.fed.type, cfg.batch_training, cfg.fed.args.epsilon)
    if cfg.fed.type == "iiadmm":
        filename = "Result_%s_%s_Batch_%s_AccumGrad_%s_CoeffGrad_%s_Penalty=%s_PrivEps_%s" % (
            DataSet_name,
            cfg.fed.type,
            cfg.batch_training,
            cfg.fed.args.accum_grad,
            cfg.fed.args.coeff_grad,
            cfg.fed.args.init_penalty,
            cfg.fed.args.epsilon,            
        )
    if cfg.fed.type == "iceadmm":
        filename = "Result_%s_%s_Batch_%s_AccumGrad_%s_Penalty=%s_PrivEps_%s" % (
            DataSet_name,
            cfg.fed.type,
            cfg.batch_training,
            cfg.fed.args.accum_grad,
            cfg.fed.args.init_penalty,
            cfg.fed.args.epsilon,
        )        

    file_ext = ".txt"
    file = dir + "/%s%s" % (filename, file_ext)
    uniq = 1
    while os.path.exists(file):
        file = dir + "/%s_%d%s" % (filename, uniq, file_ext)
        uniq += 1
    outfile = open(file, "w")
    title = "%12s %12s %12s %12s %12s %12s %12s %12s %12s %12s %12s %12s \n" % (
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
    outfile.write(title)
    print(title, end="")
    return outfile


def print_write_result_iteration(
    outfile,
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
    rho_max,
):
    results = "%12d %12.2f %12.2f %12.2f %12.2f %12.2f %12.6f %12.2f %12.4e %12.4e %12.2f %12.2f \n" % (
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
    print(results, end="")
    outfile.write(results)
    return outfile


def print_write_result_summary(
    cfg: DictConfig,
    outfile,
    comm_size,
    DataSet_name,
    num_clients,
    Elapsed_time,
    BestAccuracy,
):

    outfile.write("Device=%s \n" % (cfg.device))
    outfile.write("#Processors=%s \n" % (comm_size))
    outfile.write("Dataset=%s \n" % (DataSet_name))
    outfile.write("#Clients=%s \n" % (num_clients))
    outfile.write("Algorithm=%s \n" % (cfg.fed.type))
    outfile.write("Comm_Rounds=%s \n" % (cfg.num_epochs))
    outfile.write("Local_Epochs=%s \n" % (cfg.fed.args.num_local_epochs))
    outfile.write("DP_Eps=%s \n" % (cfg.fed.args.epsilon))    
    outfile.write("Clipping=%s \n" % (cfg.fed.args.clip_value))    
    outfile.write("Elapsed_time=%s \n" % (round(Elapsed_time, 2)))
    outfile.write("BestAccuracy=%s \n" % (round(BestAccuracy, 2)))
 

    outfile.close()
