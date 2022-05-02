from .server_federated_pca import FedServerPCA
import torch

class ServerFedAvgMomentumPCA(FedServerPCA):
    def update_global_state(self):
        
        super(ServerFedAvgMomentumPCA, self).compute_pseudo_gradient()

        super(ServerFedAvgMomentumPCA, self).update_m_vector()

        self.global_state_vec += - torch.mm( self.P.transpose(0, 1), self.m_vector.reshape(-1,1) )
         

    def logging_summary(self, cfg, logger):
        super(FedServerPCA, self).log_summary(cfg, logger)

        logger.info("client_learning_rate = %s " % (cfg.fed.args.optim_args.lr))
        logger.info(
            "server_momentum_param_1 = %s " % (cfg.fed.args.server_momentum_param_1)
        )

        if cfg.summary_file != "":
            with open(cfg.summary_file, "a") as f:

                f.write(
                    cfg.logginginfo.DataSet_name
                    + " FedAvgM ClientLR "
                    + str(cfg.fed.args.optim_args.lr)
                    + " MParam1 "
                    + str(cfg.fed.args.server_momentum_param_1)
                    + " TestAccuracy "
                    + str(cfg.logginginfo.accuracy)
                    + " BestAccuracy "
                    + str(cfg.logginginfo.BestAccuracy)
                    + " Time "
                    + str(round(cfg.logginginfo.Elapsed_time, 2))
                    + "\n"
                )
