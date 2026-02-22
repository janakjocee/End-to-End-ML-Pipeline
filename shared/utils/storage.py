"""
Storage Management Utilities
=============================

S3-compatible storage interface for model artifacts and datasets.
"""

import os
import io
import json
import pickle
from typing import Any, Optional, BinaryIO, Dict
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

from shared.utils.logger import get_logger

logger = get_logger(__name__)


class StorageManager:
    """
    S3-compatible storage manager for artifacts and datasets.
    """
    
    def __init__(
        self,
        endpoint_url: Optional[str] = None,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        region: str = 'us-east-1'
    ):
        self.endpoint_url = endpoint_url or os.getenv('MINIO_ENDPOINT_URL', 'http://localhost:9000')
        self.access_key = access_key or os.getenv('MINIO_ACCESS_KEY', 'minioadmin')
        self.secret_key = secret_key or os.getenv('MINIO_SECRET_KEY', 'minioadmin')
        self.region = region
        self._client = None
        self._resource = None
        
    def _get_client(self):
        """Get or create S3 client."""
        if self._client is None:
            self._client = boto3.client(
                's3',
                endpoint_url=self.endpoint_url,
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
                region_name=self.region
            )
        return self._client
        
    def _get_resource(self):
        """Get or create S3 resource."""
        if self._resource is None:
            self._resource = boto3.resource(
                's3',
                endpoint_url=self.endpoint_url,
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
                region_name=self.region
            )
        return self._resource
        
    def create_bucket(self, bucket_name: str) -> bool:
        """Create a new bucket."""
        try:
            client = self._get_client()
            client.create_bucket(Bucket=bucket_name)
            logger.info(f"Created bucket: {bucket_name}")
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == 'BucketAlreadyExists':
                logger.info(f"Bucket already exists: {bucket_name}")
                return True
            logger.error(f"Failed to create bucket: {str(e)}")
            raise
            
    def bucket_exists(self, bucket_name: str) -> bool:
        """Check if bucket exists."""
        try:
            client = self._get_client()
            client.head_bucket(Bucket=bucket_name)
            return True
        except ClientError:
            return False
            
    def upload_file(
        self,
        local_path: str,
        bucket_name: str,
        object_key: str,
        metadata: Optional[Dict] = None
    ) -> str:
        """Upload a file to S3."""
        try:
            client = self._get_client()
            extra_args = {}
            if metadata:
                extra_args['Metadata'] = metadata
                
            client.upload_file(local_path, bucket_name, object_key, ExtraArgs=extra_args)
            
            url = f"{self.endpoint_url}/{bucket_name}/{object_key}"
            logger.info(f"Uploaded file to: {url}")
            return url
        except ClientError as e:
            logger.error(f"Failed to upload file: {str(e)}")
            raise
            
    def upload_object(
        self,
        data: bytes,
        bucket_name: str,
        object_key: str,
        content_type: str = 'application/octet-stream',
        metadata: Optional[Dict] = None
    ) -> str:
        """Upload bytes to S3."""
        try:
            client = self._get_client()
            extra_args = {'ContentType': content_type}
            if metadata:
                extra_args['Metadata'] = metadata
                
            client.put_object(
                Bucket=bucket_name,
                Key=object_key,
                Body=data,
                **extra_args
            )
            
            url = f"{self.endpoint_url}/{bucket_name}/{object_key}"
            logger.info(f"Uploaded object to: {url}")
            return url
        except ClientError as e:
            logger.error(f"Failed to upload object: {str(e)}")
            raise
            
    def download_file(self, bucket_name: str, object_key: str, local_path: str) -> str:
        """Download a file from S3."""
        try:
            client = self._get_client()
            
            # Ensure directory exists
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            
            client.download_file(bucket_name, object_key, local_path)
            logger.info(f"Downloaded file to: {local_path}")
            return local_path
        except ClientError as e:
            logger.error(f"Failed to download file: {str(e)}")
            raise
            
    def download_object(self, bucket_name: str, object_key: str) -> bytes:
        """Download object as bytes."""
        try:
            client = self._get_client()
            response = client.get_object(Bucket=bucket_name, Key=object_key)
            return response['Body'].read()
        except ClientError as e:
            logger.error(f"Failed to download object: {str(e)}")
            raise
            
    def delete_object(self, bucket_name: str, object_key: str) -> bool:
        """Delete an object from S3."""
        try:
            client = self._get_client()
            client.delete_object(Bucket=bucket_name, Key=object_key)
            logger.info(f"Deleted object: {object_key}")
            return True
        except ClientError as e:
            logger.error(f"Failed to delete object: {str(e)}")
            raise
            
    def list_objects(self, bucket_name: str, prefix: str = '') -> list:
        """List objects in a bucket."""
        try:
            client = self._get_client()
            response = client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
            return [obj['Key'] for obj in response.get('Contents', [])]
        except ClientError as e:
            logger.error(f"Failed to list objects: {str(e)}")
            raise
            
    def save_model(
        self,
        model: Any,
        bucket_name: str,
        model_name: str,
        version: str,
        metadata: Optional[Dict] = None
    ) -> str:
        """Save a model to S3."""
        import pickle
        
        model_key = f"models/{model_name}/{version}/model.pkl"
        model_bytes = pickle.dumps(model)
        
        full_metadata = {
            'model_name': model_name,
            'version': version,
            'framework': 'sklearn',
            **(metadata or {})
        }
        
        return self.upload_object(
            model_bytes,
            bucket_name,
            model_key,
            content_type='application/octet-stream',
            metadata=full_metadata
        )
        
    def load_model(self, bucket_name: str, model_name: str, version: str) -> Any:
        """Load a model from S3."""
        import pickle
        
        model_key = f"models/{model_name}/{version}/model.pkl"
        model_bytes = self.download_object(bucket_name, model_key)
        return pickle.loads(model_bytes)
        
    def save_dataset(
        self,
        df,
        bucket_name: str,
        dataset_name: str,
        version: str,
        format: str = 'parquet'
    ) -> str:
        """Save a pandas DataFrame to S3."""
        import pandas as pd
        
        buffer = io.BytesIO()
        
        if format == 'parquet':
            df.to_parquet(buffer, index=False)
            content_type = 'application/octet-stream'
        elif format == 'csv':
            df.to_csv(buffer, index=False)
            content_type = 'text/csv'
        else:
            raise ValueError(f"Unsupported format: {format}")
            
        buffer.seek(0)
        
        dataset_key = f"datasets/{dataset_name}/{version}/data.{format}"
        return self.upload_object(
            buffer.getvalue(),
            bucket_name,
            dataset_key,
            content_type=content_type
        )
        
    def load_dataset(
        self,
        bucket_name: str,
        dataset_name: str,
        version: str,
        format: str = 'parquet'
    ):
        """Load a pandas DataFrame from S3."""
        import pandas as pd
        
        dataset_key = f"datasets/{dataset_name}/{version}/data.{format}"
        data_bytes = self.download_object(bucket_name, dataset_key)
        
        buffer = io.BytesIO(data_bytes)
        
        if format == 'parquet':
            return pd.read_parquet(buffer)
        elif format == 'csv':
            return pd.read_csv(buffer)
        else:
            raise ValueError(f"Unsupported format: {format}")
            
    def get_object_metadata(self, bucket_name: str, object_key: str) -> Dict:
        """Get object metadata."""
        try:
            client = self._get_client()
            response = client.head_object(Bucket=bucket_name, Key=object_key)
            return {
                'content_type': response.get('ContentType'),
                'content_length': response.get('ContentLength'),
                'last_modified': response.get('LastModified'),
                'metadata': response.get('Metadata', {})
            }
        except ClientError as e:
            logger.error(f"Failed to get object metadata: {str(e)}")
            raise