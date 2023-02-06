from appfl.funcx.cloud_storage import CloudStorage
from omegaconf import OmegaConf

CloudStorage.init(OmegaConf.create({"bucket": "anl-appfl"}))

cs = CloudStorage.get_instance()
print(cs.upload("./funcx_async.py", "funcx_async"))
cs.download("funcx_async", "test.txt")
