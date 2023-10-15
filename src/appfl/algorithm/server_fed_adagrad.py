import torch
from .server_federated import FedServer

class ServerFedAdagrad(FedServer):
    def compute_step(self):
        super(ServerFedAdagrad, self).compute_pseudo_gradient()
        super(ServerFedAdagrad, self).update_m_vector()
        for name, _ in self.model.named_parameters():
            self.v_vector[name] = self.v_vector[name] + torch.square(-self.pseudo_grad[name])
            self.step[name] = torch.div(
                self.server_learning_rate * self.m_vector[name],
                torch.sqrt(self.v_vector[name]) + self.server_adapt_param,
            )

    def logging_summary(self, cfg, logger):
        super(FedServer, self).log_summary(cfg, logger)
        logger.info("client_learning_rate = %s " % (cfg.fed.args.optim_args.lr))
        logger.info("server_learning_rate = %s " % (cfg.fed.args.server_learning_rate))
        logger.info("server_adapt_param = %s " % (cfg.fed.args.server_adapt_param))
        logger.info("server_momentum_param_1 = %s " % (cfg.fed.args.server_momentum_param_1))
        if cfg.summary_file != "":
            with open(cfg.summary_file, "a") as f:
                f.write(
                    cfg.logginginfo.DataSet_name
                    + " FedAdagrad ClientLR "
                    + str(cfg.fed.args.optim_args.lr)
                    + " MParam1 "
                    + str(cfg.fed.args.server_momentum_param_1)
                    + " AdaptParam "
                    + str(cfg.fed.args.server_adapt_param)
                    + " ServerLR "
                    + str(cfg.fed.args.server_learning_rate)
                    + " TestAccuracy "
                    + str(cfg.logginginfo.accuracy)
                    + " BestAccuracy "
                    + str(cfg.logginginfo.BestAccuracy)
                    + " Time "
                    + str(round(cfg.logginginfo.Elapsed_time, 2))
                    + "\n"
                )
