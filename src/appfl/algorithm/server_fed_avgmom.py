from .server_federated import FedServer


class ServerFedAvgMomentum(FedServer):
    def compute_step(self):
        super(ServerFedAvgMomentum, self).compute_pseudo_gradient()
        super(ServerFedAvgMomentum, self).update_m_vector()
        for name, _ in self.model.named_parameters():
            self.step[name] = -self.m_vector[name]

    def logging_summary(self, cfg, logger):
        super(FedServer, self).log_summary(cfg, logger)

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
