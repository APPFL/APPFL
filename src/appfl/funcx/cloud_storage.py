from ..misc import mLogging
import boto3
from botocore.exceptions import ClientError
import os.path as osp

class CloudStorage(object):
    instc  = None
    def __init__(self):
        raise RuntimeError("Call get_instance() instead")
    
    @classmethod
    def get_instance(cls):
        if cls.instc is not None:
            return cls.instc
        else:
            raise RuntimeError("Please call CloudStorage.init(cfg) first")
    
    @classmethod
    def init(cls, cfg):
        if cls.instc is None:
            new_inst = cls.__new__(cls)
            new_inst.bucket = cfg.s3_bucket
            new_inst.client = boto3.client('s3')
            # new_inst.logging = mLogging.get_logger()
            cls.instc = new_inst
        return cls.instc
    
    @staticmethod
    def is_cloud_storage_object(obj):
        if type(obj) != dict:
            return False
        if 's3' in obj:
            return True
        else:
            return False
    
    @staticmethod
    def get_cloud_object_info(obj):
        if CloudStorage.is_cloud_storage_object(obj):
            return obj['s3']['bucket'], obj['s3']['object_name'], obj['s3']['file_name']
        else:
            return None

    def __get_data_address(self,file_name, object_name):
        return {
            's3':{
                'bucket': self.bucket,
                'object_name': object_name,
                'file_name': osp.basename(file_name)
            }
        }

    def __upload_file(self, file_name, object_name):
        try:
            resp = self.client.upload_file(file_name, self.bucket, object_name)
        except ClientError as e:
            return None
        return self.__get_data_address(file_name, object_name)
    
    def __download_file(self, file_name, object_name):
        self.client.download_file(self.bucket, object_name, file_name)

    def upload(self, file_path, object_name=None):
        if object_name is None:
            object_name = osp.basename(file_path)
        return self.__upload_file(file_path, object_name)

    def download(self, object_name, file_path):
        return self.__download_file(file_path, object_name)