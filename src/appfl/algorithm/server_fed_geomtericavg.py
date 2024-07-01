from .server_federated import FedServer

class ServerGeometricMean(FedServer):
    def compute_step(self):
        super(ServerGeometricMean, self).compute_geometricmean()
        for name, _ in self.model.named_parameters():
            self.step[name] = self.gm_step[name]

    def logging_summary(self, cfg, logger):
        super(ServerGeometricMean, self).log_summary(cfg, logger)
        logger.info(f"client_learning_rate = {1.0}")

        if cfg.summary_file != "":
            with open(cfg.summary_file, "a") as f:
                f.write(
                    cfg.logginginfo.DataSet_name
                    + " GeometricMean Client Initial LR "
                    + str(1.0)
                    + " TestAccuracy "
                    + str(cfg.logginginfo.accuracy)
                    + " BestAccuracy "
                    + str(cfg.logginginfo.BestAccuracy)
                    + " Time "
                    + str(round(cfg.logginginfo.Elapsed_time, 2))
                    + "\n"
                )
