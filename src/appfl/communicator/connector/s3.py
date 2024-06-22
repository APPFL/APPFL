"""S3 connector implementation."""

from __future__ import annotations

import logging
import csv
import sys
import uuid
import boto3
import requests
from types import TracebackType
from typing import Any, NamedTuple, Sequence, Optional

if sys.version_info >= (3, 11):  # pragma: >=3.11 cover
    from typing import Self
else:  # pragma: <3.11 cover
    from typing_extensions import Self

logger = logging.getLogger(__name__)

class S3Key(NamedTuple):
    """Key to objects in a the S3 bucket.

    Attributes:
        object_name: Unique object name.
        presigned_url: A presigned URL for downloading the object.
    """

    object_name: str
    presigned_url: str


class S3Connector:
    """Connector to AWS S3 bucket.

    This connector writes object to an AWS S3 bucket, and generates a
    presigned URL for downloading the object.

    Args:
        s3_bucket: Name of the S3 bucket.
        s3_creds_file: Path to a CSV file containing the AWS credentials.
        expiration: Time in seconds for the presigned URL to expire.
        clear: Clear all uploaded objects on `close()`.
    """

    def __init__(
        self, 
        s3_bucket: str,
        expiration: int = 3600,
        s3_creds_file: Optional[str] = None,
        clear: bool = True,
    ) -> None:
        logger.debug(f"Creating S3 connector for bucket {s3_bucket}")
        self.bucket = s3_bucket
        s3_kwargs = {}
        if s3_creds_file is not None:
            with open(s3_creds_file) as file:
                reader = csv.reader(file)
                keys = next(reader)
                s3_kwargs = {
                    'region_name': keys[0],
                    'aws_access_key_id': keys[1],
                    'aws_secret_access_key': keys[2]
                }
        # Alternatively, you can provide the credentials via command line
        # This only needs to be done once.
        # (1) Install the AWS CLI `pip install awscli`
        # (2) Run `aws configure` and provide the credentials
        self.client = boto3.client('s3', **s3_kwargs)
        self.uploaded_obj = set()
        self.expiration = expiration
        self.clear = clear

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        exc_traceback: TracebackType | None,
    ) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}(s3_bucket={self.bucket}, '
            f'expiration={self.expiration})'
        )

    def close(self, clear: bool | None = None) -> None:
        """Close the connector and clean up.

        Warning:
            This method should only be called at the end of the program
            when the connector will no longer be used, for example once all
            proxies have been resolved.

        Args:
            clear: Remove the store directory. Overrides the default
                value of `clear` provided when the `S3Connector` was instantiated.
        """
        for obj in self.uploaded_obj:
            try:
                self.client.delete_object(Bucket=self.bucket, Key=obj.object_name)
                logger.debug(f"Deleted object {obj.object_name}")
            except:
                logger.error(f"Failed to delete object {obj.object_name}")

    def config(self) -> dict[str, Any]:
        """Get the connector configuration.

        The configuration contains all the information needed to reconstruct
        the connector object.
        """
        return {
            's3_bucket': self.bucket,
            'expiration': self.expiration,
            'clear': self.clear,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> S3Connector:
        """Create a new connector instance from a configuration.

        Args:
            config: Configuration returned by `#!python .config()`.
        """
        return cls(**config)

    def evict(self, key: S3Key) -> None:
        """Evict the object associated with the key.

        Args:
            key: Key associated with object to evict.
        """
        self.client.delete_object(Bucket=self.bucket, Key=key.object_name)

    def exists(self, key: S3Key) -> bool:
        """Check if an object associated with the key exists.

        Args:
            key: Key potentially associated with stored object.

        Returns:
            If an object associated with the key exists.
        """
        try:
            response = requests.head(key.presigned_url)
            if response.status_code == 200:
                return True
            return False
        except:
            return False

    def get(self, key: S3Key) -> bytes | None:
        """Get the serialized object associated with the key.

        Args:
            key: Key associated with the object to retrieve.

        Returns:
            Serialized object or `None` if the object does not exist.
        """
        try:
            response = requests.get(key.presigned_url)
            if response.status_code == 200:
                return response.content
            return None
        except:
            return None

    def get_batch(self, keys: Sequence[S3Key]) -> list[bytes | None]:
        """Get a batch of serialized objects associated with the keys.

        Args:
            keys: Sequence of keys associated with objects to retrieve.

        Returns:
            List with same order as `keys` with the serialized objects or
            `None` if the corresponding key does not have an associated object.
        """
        return [self.get(key) for key in keys]

    def put(self, obj: bytes) -> S3Key:
        """Put a serialized object in the store.

        Args:
            obj: Serialized object to put in the store.

        Returns:
            Key which can be used to retrieve the object.
        """
        object_name = str(uuid.uuid4())

        self.client.put_object(Bucket=self.bucket, Key=object_name, Body=obj)
        presigned_url = self.client.generate_presigned_url(
            'get_object',
            Params={'Bucket': self.bucket, 'Key': object_name},
            ExpiresIn=self.expiration,
        )

        key = S3Key(
            object_name=object_name,
            presigned_url=presigned_url
        )

        self.uploaded_obj.add(key)
        return key

    def put_batch(self, objs: Sequence[bytes]) -> list[S3Key]:
        """Put a batch of serialized objects in the store.

        Args:
            objs: Sequence of serialized objects to put in the store.

        Returns:
            List of keys with the same order as `objs` which can be used to
            retrieve the objects.
        """
        return [self.put(obj) for obj in objs]
