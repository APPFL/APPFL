import boto3
import os
import os.path as osp
import sys
import time

from botocore.exceptions import ClientError

from appfl.misc import dump_data_to_file, load_data_from_file, id_generator


class LargeObjectWrapper(object):
    MAX_SIZE_LIMIT = 1 * 1024 * 1024
    DEBUG = True

    def __init__(self, data, name: str):
        self.data = data
        self.name = name

    @property
    def size(self):
        return sys.getsizeof(self.data)

    @property
    def can_send_directly(self):
        if LargeObjectWrapper.DEBUG:
            return False
        return self.size < LargeObjectWrapper.MAX_SIZE_LIMIT


class CloudStorage(object):
    instc = None

    def __init__(self):
        raise RuntimeError("Call get_instance() instead")

    @classmethod
    def get_instance(cls):
        if cls.instc is not None:
            return cls.instc
        else:
            raise RuntimeError("Please call CloudStorage.init(cfg) first")

    @classmethod
    def init(cls, cfg, temp_dir="./tmp", logger=None):
        if cls.instc is None:
            new_inst = cls.__new__(cls)
            new_inst.bucket = cfg.server.s3_bucket
            new_inst.client = boto3.client("s3")
            new_inst.temp_dir = temp_dir
            new_inst.logger = logger
            new_inst.session_id = id_generator() + str(time.time())
            cls.instc = new_inst
        return cls.instc

    @staticmethod
    def is_cloud_storage_object(obj):
        if type(obj) != dict:
            return False
        if "s3" in obj:
            return True
        else:
            return False

    @staticmethod
    def get_cloud_object_info(obj):
        if CloudStorage.is_cloud_storage_object(obj):
            return obj["s3"]["bucket"], obj["s3"]["object_name"], obj["s3"]["file_name"]
        else:
            return None

    def __get_data_address(self, file_name, object_name):
        return {
            "s3": {
                "bucket": self.bucket,
                "object_name": object_name,
                "file_name": osp.basename(file_name),
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

    def upload_file(self, file_path: str, object_name: str = None):
        if object_name is None:
            object_name = self.instc.session_id + "_" + osp.basename(file_path)
        return self.__upload_file(file_path, object_name)

    def download_file(self, object_name, file_path):
        return self.__download_file(file_path, object_name)

    @classmethod
    def upload_object(cls, data, object_name=None, ext="pkl", temp_dir=None):
        if object_name is None:
            assert type(data) == LargeObjectWrapper
            object_name = data.name
            data = data.data

        assert ext in ["pkl", "pt", "pth"]
        cs = cls.get_instance()

        # Prepare temp dir
        temp_dir = temp_dir if temp_dir is not None else cs.temp_dir
        os.makedirs(temp_dir, exist_ok=True)
        file_path = osp.join(temp_dir, "%s.%s" % (object_name, ext))

        # Save data to file
        dump_data_to_file(data, file_path)

        # Logging
        if cs.logger is not None:
            file_size = osp.getsize(file_path) * 1e-3
            cs.logger.info(
                "Uploading object '%s' (%.01f KB) to S3" % (object_name, file_size)
            )
        # Upload file
        return cs.upload_file(file_path)

    @classmethod
    def download_object(cls, data_info: dict, temp_dir: str = None, to_device=None):
        cs = cls.get_instance()
        # Prepare temp_dir
        _, object_name, file_name = cls.get_cloud_object_info(data_info)
        temp_dir = temp_dir if temp_dir is not None else cs.temp_dir
        os.makedirs(temp_dir, exist_ok=True)
        file_path = osp.join(temp_dir, file_name)

        # Download file
        cs.download_file(object_name, file_path)

        # Logging
        if cs.logger is not None:
            file_size = osp.getsize(file_path) * 1e-3
            cs.logger.info(
                "Downloaded object '%s' (%.01f) from S3" % (object_name, file_size)
            )
        return load_data_from_file(file_path, to_device)

    @classmethod
    def clean_up(self):
        """Clean up files on cloud storage"""
        pass
