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
