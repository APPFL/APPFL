from appfl.algorithm import *
from appfl.funcx.cloud_storage import LargeObjectWrapper
from appfl.funcx.funcx_sync_server import APPFLFuncXSyncServer


class APPFLFuncXTestServer(APPFLFuncXSyncServer):
    def _run_sync_task(self, func, *args, **kwargs):
        _args = [
            args[i].data if type(args[i]) == LargeObjectWrapper else args[i]
            for i in range(len(args))
        ]
        _kwargs = {
            k: kwargs[k].data if type(kwargs[k]) == LargeObjectWrapper else kwargs[k]
            for k in kwargs
        }
        client_results = OrderedDict()
        client_logs = OrderedDict()
        self.cfg.use_cloud_transfer = False

        res = func(self.cfg, 0, *_args, **_kwargs)
        try: 
            len(res)
        except:
            client_results[0] = res
            client_logs = None
        else:
            client_results[0] = res[0]
            client_logs[0] = res[1]
        return client_results, client_logs
