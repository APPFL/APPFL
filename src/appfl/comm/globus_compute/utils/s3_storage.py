import os
import csv
import sys
import time
import boto3
import requests
import os.path as osp
from botocore.exceptions import ClientError
from appfl.misc.utils import dump_data_to_file, load_data_from_file, id_generator

class LargeObjectWrapper(object):
    MAX_SIZE_LIMIT = 1 * 1      # anything using this wrapper is sent via S3
    def __init__(self, data, name: str):
        self.data = data
        self.name= name
    
    @property
    def size(self):
        return sys.getsizeof(self.data)
    
    @property
    def can_send_directly(self):
        return self.size < LargeObjectWrapper.MAX_SIZE_LIMIT

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
    def init(cls, cfg, temp_dir="./tmp", logger = None):
        if cls.instc is None:
            new_inst = cls.__new__(cls)
            new_inst.bucket = cfg.server.s3_bucket
            kwargs = {}
            if cfg.server.s3_creds != "":
                with open(cfg.server.s3_creds) as file:
                    reader = csv.reader(file)
                    keys = next(reader)
                    kwargs = {
                        'region_name': keys[0],
                        'aws_access_key_id': keys[1],
                        'aws_secret_access_key': keys[2]
                    }     
            new_inst.client = boto3.client(
                service_name='s3',
                **kwargs
            )
            new_inst.temp_dir= temp_dir
            new_inst.logger = logger
            new_inst.session_id = id_generator() + str(time.time())
            new_inst.registered_obj = set()
            new_inst.uploaded_obj = {}
            cls.instc = new_inst
        return cls.instc
    
    @staticmethod
    def is_cloud_storage_object(obj):
        """Check if the object corresponds to an object stored on the cloud (S3 bucket)"""
        if type(obj) != dict:
            return False
        if 's3' in obj:
            return True
        else:
            return False
        
    @staticmethod
    def get_cloud_object_info(obj):
        """Obtain the storage information for an object stored on S3 bucket"""
        file_name = obj['s3']['file_name']
        object_name = obj['s3']['object_name'] if 'object_name' in obj['s3'] else None
        object_url = obj['s3']['object_url'] if 'object_url' in obj['s3'] else None
        return file_name, object_name, object_url

        
    def upload_file(self, file_path: str, object_url: str = None, object_name: str = None, expiration: int = 3600, delete_local: bool = True):
        """
        Upload a local file to S3 directly via object_name or using the presigned url
        Inputs:
            file_path: path to local file
            object_name: object key for the file on S3 bucket
            expiration: expiration second for the returned presigned url
            delete_local: whether to delete the local file
        Output:
            s3_obj: an object containing the following informaiton
                - if uploaded directly: file name, presigned download url
                - if uploaded via presigned url: file name, object name
        """
        # Upload the object using presigned url
        if object_url is not None:
            try:
                with open(file_path, 'rb') as f:
                    response = requests.put(object_url, data=f)
                if delete_local:
                    os.remove(file_path)
            except:
                raise
            s3_obj = {
                's3': {
                    'file_name': osp.basename(file_path),
                    'object_name': object_name
                }
            }
            self.uploaded_obj[object_name] = s3_obj
            return s3_obj
        # Upload the object directly
        try:
            self.client.upload_file(file_path, self.bucket, object_name)
            if delete_local:
                os.remove(file_path)
        except ClientError as e:
            if self.logger is not None:
                self.logger.info(f"Error occurs in uploading file {e}")
            else:
                print(f"Error occurs in uploading file {e}")
            raise
        try: 
            response = self.client.generate_presigned_url(
                'get_object',
                Params = {'Bucket': self.bucket, 'Key': object_name},
                ExpiresIn = expiration
            )
            s3_obj = {
                's3': {
                    'file_name': osp.basename(file_path),
                    'object_url': response
                }
            }
            self.uploaded_obj[object_name] = s3_obj
            return s3_obj
        except Exception as e:
            if self.logger is not None:
                self.logger.info(f"Error occurs in generating presigned url {e}")
            else:
                print(f"Error occurs in generating presigned url {e}")
            raise

    def download_file(self, file_path, object_name = None, object_url = None, delete_cloud = False):
        """
        Download an object from S3 bucket either using the object name (only possible with credentials)
        or using a presigned object url.
        Inputs:
            file_path: path to store the downloaded file
            object_name: key of the object on S3 to download
            object_url: presigned url for downloading the file
            delete_cloud: delete the cloud object after downloading using object name (only possible with credentials)
        """
        if object_url is not None:
            try:
                response = requests.get(object_url)
                if response.status_code == 200:
                    with open(file_path, "wb") as file:
                        file.write(response.content)
                    if self.logger is not None:
                        self.logger.info(f"Successfully donwload object using presigned url")
                    else:
                        print(f"Successfully donwload object using presigned url")
                return
            except:
                if self.logger is not None:
                    self.logger.info(f"Error in downloading object using presigned url")
                else:
                    print(f"Error in downloading object using presigned url")
                raise
        if object_name is not None:
            try:
                self.client.download_file(self.bucket, object_name, file_path)
            except:
                if self.logger is not None:
                    self.logger.info(f"Error in downloading object {object_name} from S3")
                else:
                    print(f"Error in downloading object {object_name} from S3")
                raise
            try:
                if delete_cloud:
                    self.client.delete_object(Bucket=self.bucket, Key=object_name)
                return
            except:
                if self.logger is not None:
                    self.logger.info(f"Error in deleteing object {object_name} from S3")
                else:
                    print(f"Error in deleteing object {object_name} from S3")

    @classmethod
    def upload_object(cls, data, object_name = None, object_url = None, ext = 'pt', temp_dir = None, register_for_clean = False):
        """
        Upload a python object to S3 bucket by first saving the object into a file in the temp directory,
        and then uploading to S3 directly via object name (only with credentials) or via presigned url.
        Inputs:
            cls: the cloud storage class instance
            data: the python object to be uploaded to S3 bucket
            object_name: key of the object to be uploaded to S3 bucket
            object_url: presigned url for uploading the object to S3 bucket
            ext: extension of the file for saving the object
            temp_dir: temporary directory for storing the saved object
            register_for_clean: if True, record the name of the object which will be cleaned in the cleanup function
        Outputs:
            s3_obj: an object containing the following informaiton
                - if uploaded directly: file name, presigned download url
                - if uploaded via presigned url: file name, object name
        """
        if object_name is None:
            assert type(data) == LargeObjectWrapper
            object_name = data.name
            data = data.data
        cs = cls.get_instance()

        if object_name in cs.uploaded_obj:
            return cs.uploaded_obj[object_name]

        temp_dir = temp_dir if temp_dir is not None else cs.temp_dir
        os.makedirs(temp_dir, exist_ok=True)
        file_path = osp.join(temp_dir, "%s.%s" % (object_name, ext))

        dump_data_to_file(data, file_path)

        file_size = osp.getsize(file_path)
        size_metric = ['KB', 'MB', 'GB']
        for metric in size_metric:
            file_size *= 1e-3
            if file_size < 1000 or metric == 'GB':
                if cs.logger is not None:
                    cs.logger.info(f"Uploading model ({file_size:.2f} {metric}) to S3")
                else:
                    print(f"Uploading model ({file_size:.2f} {metric}) to S3")
                break

        if object_name is not None and register_for_clean:
            cs.registered_obj.add(object_name)

        return cs.upload_file(file_path, object_url, object_name)
    
    @classmethod
    def download_object(cls, data_info: dict, temp_dir: str = None, to_device: str = "cpu", delete_cloud: bool = False, delete_local: bool = True):
        """
        Download a python object from S3 either using the object_name (only possible with credentials)
        or using the presigned url, save the object into the temporary directory, and load into python object.
        Inputs:
            cls: the cloud storage class instance
            data_info: information about the cloud storage object
            temp_dir: temporary directory for storing the downloaded object
            to_device: the device of the downloaded object (only applicable to PyTorch model)
            delete_cloud: whether to delete the object stored on S3 after downloading the object
            delete_local: whether to delete the local file used for temporary storage of the object
        Output:
            object: the downloaded object
        """
        cs = cls.get_instance()
        file_name, object_name, object_url = cls.get_cloud_object_info(data_info)

        temp_dir = temp_dir if temp_dir is not None else cs.temp_dir
        os.makedirs(temp_dir, exist_ok=True)
        file_path = osp.join(temp_dir, file_name)

        cs.download_file(file_path, object_name, object_url, delete_cloud)

        file_size = osp.getsize(file_path)
        size_metric = ['KB', 'MB', 'GB']
        for metric in size_metric:
            file_size *= 1e-3
            if file_size < 1000 or metric == 'GB':
                if cs.logger is not None:
                    cs.logger.info(f"Downloaded model ({file_size:.2f} {metric}) from S3")
                else:
                    print(f"Downloaded model ({file_size:.2f} {metric}) from S3")
                break

        data = load_data_from_file(file_path, to_device)
        
        if delete_local:
            try:
                os.remove(file_path)
            except:
                pass
        
        return data
    
    @classmethod
    def presign_upload_object(cls, object_name: str, expiration: int = 3600):
        """Presign a url for uploading an object to S3 bucket"""
        cs = cls.get_instance()
        try:
            response = cs.client.generate_presigned_url(
                'put_object',
                Params={'Bucket': cs.bucket, 'Key': object_name},
                ExpiresIn=expiration
            )
        except Exception as e:
            if cs.logger is not None:
                cs.logger.info(f"Error in generating presigned url for uploading object {e}")
            else:
                print(f"Error in generating presigned url for uploading object {e}")
            raise
        return response

    @classmethod
    def clean_up(cls):
        """Clean up registered objects on S3 bucket"""
        cs = cls.get_instance()
        for obj in cs.registered_obj:
            try:
                cs.client.delete_object(Bucket=cs.bucket, Key=obj)
                if cs.logger is not None:
                    cs.logger.info(f"Successfully cleaned object {obj} on S3")
                else:
                    print(f"Successfully cleaned object {obj} on S3")
            except Exception as e:
                if cs.logger is not None:
                    cs.logger.info(f"Error in cleaning object {obj} on S3 {e}")
                else:
                    print(f"Error in cleaning object {obj} on S3 {e}")