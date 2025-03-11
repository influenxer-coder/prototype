import os

import boto3
from botocore.exceptions import ClientError

from app.config.settings import Config


class S3Service:
    def __init__(self):
        self.client = boto3.client('s3',
                                   aws_access_key_id=Config.AWS_ACCESS_KEY,
                                   aws_secret_access_key=Config.AWS_SECRET_KEY,
                                   region_name=Config.AWS_REGION)

    def exists_in_bucket(self, bucket_name: str, filename: str) -> bool:
        try:
            self.client.head_object(Bucket=bucket_name, Key=filename)
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return False
            else:
                raise e

    def upload_to_s3(self, bucket_name: str, filename: str, temp_file: str) -> str | None:
        try:
            object_name = f"{filename}"

            self.client.upload_file(temp_file, bucket_name, object_name)
            s3_location = f"s3://{bucket_name}/{object_name}"
            print(f"s3_location = {s3_location}")
            return s3_location
        except Exception as e:
            print(f"An error occurred while uploading the video to s3 {filename}: {str(e)}")
            return None

    # def download_from_s3(self, s3_url: str, local_path: str) -> bool:
    #     try:
    #         # Parse the S3 URL
    #         bucket_name = s3_url.split("/")[2]
    #         key = "/".join(s3_url.split("/")[3:])
    #
    #         # Download the file
    #         self.client.download_file(bucket_name, key, local_path)
    #         return True
    #     except Exception as e:
    #         print(f"Error downloading {s3_url}: {e}")
    #         return False

    def download_from_s3(self, s3_url: str, local_path: str) -> bool:
        try:
            # Parse the S3 URL
            bucket_name = s3_url.split("/")[2]
            key = "/".join(s3_url.split("/")[3:])

            # Make sure the directory exists
            os.makedirs(os.path.dirname(local_path), exist_ok=True)

            # Download the file
            print(f"Downloading from {bucket_name}/{key} to {local_path}")
            self.client.download_file(bucket_name, key, local_path)
            return True
        except Exception as e:
            print(f"Error downloading file from S3: {str(e)}")
            print(f"URL: {s3_url}, Local Path: {local_path}")
            return False
