import os
from omegaconf import DictConfig

def print_write_result_title(cfg: DictConfig, DataSet_name: str ):
    ## Print and Write Results  
    dir = cfg.result_dir
    if os.path.isdir(dir) == False:
        os.mkdir(dir)            
    filename = "Result_%s_%s"%(DataSet_name, cfg.fed.type)    
    if cfg.fed.type == "iadmm":  
        filename = "Result_%s_%s(rho=%s)"%(DataSet_name, cfg.fed.type, cfg.fed.args.penalty)
    
    file_ext = ".txt"
    file = dir+"/%s%s"%(filename,file_ext)
    uniq = 1
    while os.path.exists(file):
        file = dir+"/%s_%d%s"%(filename, uniq, file_ext)
        uniq += 1
    outfile = open(file,"w")
    title = (
            "%12s %12s %12s %12s %12s %12s %12s \n"
            % (
                "Iter",                
                "Local[s]",
                "Global[s]",
                "Iter[s]",
                "Elapsed[s]",
                "TestAvgLoss",
                "TestAccuracy"                
            )
        )    
    outfile.write(title)
    print(title, end="")
    return outfile


def print_write_result_iteration(outfile, t, LocalUpdate_time, GlobalUpdate_time, PerIter_time, Elapsed_time,test_loss,accuracy):    
    results = (
                "%12d %12.2f %12.2f %12.2f %12.2f %12.6f %12.2f \n"
                % (
                    t+1,
                    LocalUpdate_time,
                    GlobalUpdate_time,
                    PerIter_time,
                    Elapsed_time,
                    test_loss,
                    accuracy                 
                )
            )        
    print(results, end="")
    outfile.write(results)
    return outfile


def print_write_result_summary(cfg: DictConfig, outfile, comm_size, DataSet_name, num_clients, Elapsed_time, BestAccuracy):

    outfile.write("Device=%s \n"%(cfg.device))
    outfile.write("#Processors=%s \n"%(comm_size))
    outfile.write("Dataset=%s \n"%(DataSet_name))
    outfile.write("#Clients=%s \n"%(num_clients))        
    outfile.write("Algorithm=%s \n"%(cfg.fed.type))
    outfile.write("Comm_Rounds=%s \n"%(cfg.num_epochs))
    outfile.write("Local_Epochs=%s \n"%(cfg.fed.args.num_local_epochs))    
    outfile.write("Elapsed_time=%s \n"%(round(Elapsed_time,2)))  
    outfile.write("BestAccuracy=%s \n"%(round(BestAccuracy,2))      
    
    if cfg.fed.type == "iadmm":
        outfile.write("ADMM Penalty=%s \n"%(cfg.fed.args.penalty))

    outfile.close()
