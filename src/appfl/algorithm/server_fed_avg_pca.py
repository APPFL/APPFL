from .server_federated_pca import FedServerPCA
import torch

class ServerFedAvgPCA(FedServerPCA):

    def update_global_state(self):

        super(ServerFedAvgPCA, self).compute_pseudo_gradient()
        
        self.global_state_vec += - torch.mm( self.P.transpose(0, 1), self.pseudo_grad.reshape(-1,1) )



    def logging_summary(self, cfg, logger):
        
        super(FedServerPCA, self).log_summary(cfg, logger)

        logger.info("client_learning_rate = %s " % (cfg.fed.args.optim_args.lr))

        if cfg.summary_file != "":
            with open(cfg.summary_file, "a") as f:

                f.write(
                    cfg.logginginfo.DataSet_name
                    + " FedAvg ClientLR "
                    + str(cfg.fed.args.optim_args.lr)
                    + " TestAccuracy "
                    + str(cfg.logginginfo.accuracy)
                    + " BestAccuracy "
                    + str(cfg.logginginfo.BestAccuracy)
                    + " Time "
                    + str(round(cfg.logginginfo.Elapsed_time, 2))
                    + "\n"
                )
