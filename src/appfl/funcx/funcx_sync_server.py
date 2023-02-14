from omegaconf import DictConfig
from funcx import FuncXClient


from appfl.algorithm import *
from appfl.funcx.cloud_storage import LargeObjectWrapper
from appfl.funcx.funcx_server import APPFLFuncXServer
from appfl.funcx.funcx_client import client_training


class APPFLFuncXSyncServer(APPFLFuncXServer):
    def __init__(self, cfg: DictConfig, fxc: FuncXClient):
        super(APPFLFuncXSyncServer, self).__init__(cfg, fxc)
        cfg["logginginfo"]["comm_size"] = 1

    def _do_training(self):
        """Looping over all epochs"""
        start_time = time.time()
        for t in range(self.cfg.num_epochs):
            self.logger.info(
                " ====== Epoch [%d/%d] ====== " % (t + 1, self.cfg.num_epochs)
            )
            per_iter_start = time.time()
            """ Do one training steps"""
            """ Training """
            ## Get current global state
            global_state = self.server.model.state_dict()
            local_update_start = time.time()
            ## Perform LR decay
            self._lr_step(t)
            ## Boardcast global state and start training at funcX endpoints and aggregate local updates from clients
            local_states, client_logs = self._run_sync_task(
                client_training,
                self.weights,
                LargeObjectWrapper(global_state, "server_state"),
                self.loss_fn,
                do_validation=self.cfg.client_do_validation,
            )

            local_states = [local_states]
            self._do_client_validation(t, client_logs)
            self.cfg["logginginfo"]["LocalUpdate_time"] = (
                time.time() - local_update_start
            )

            ## Perform global update
            global_update_start = time.time()
            self.server.update(local_states)
            self.cfg["logginginfo"]["GlobalUpdate_time"] = (
                time.time() - global_update_start
            )
            self.cfg["logginginfo"]["PerIter_time"] = time.time() - per_iter_start
            self.cfg["logginginfo"]["Elapsed_time"] = time.time() - start_time

            """ Validation """
            if (t + 1) % self.cfg.server_validation_step == 0:
                self._do_server_validation(t + 1)

            self.server.logging_iteration(self.cfg, self.logger, t)
            """ Saving checkpoint """
            self._save_checkpoint(t)
