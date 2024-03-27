# s3_utils.py

import boto3
import os

def upload_folder_to_s3(folder_path, bucket_name, prefix=''):
    """
    Upload a folder and its contents to an S3 bucket
    
    :param folder_path: Path to the folder to upload
    :param bucket_name: Name of the S3 bucket
    :param prefix: S3 key prefix (optional)
    :return: True if folder was uploaded, else False
    """
    # Create an S3 client
    s3 = boto3.client('s3')

    try:
        # Walk through the directory
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                # Construct the full local path
                local_path = os.path.join(root, file)
                # Construct the full S3 path
                s3_path = os.path.join(prefix, os.path.relpath(local_path, folder_path))
                # Upload the file
                s3.upload_file(local_path, bucket_name, s3_path)
    except Exception as e:
        print(f"Error uploading folder: {e}")
        return False
    return True

def download_folder_from_s3(bucket_name, prefix, local_path):
    """
    Download a folder and its contents from an S3 bucket
    
    :param bucket_name: Name of the S3 bucket
    :param prefix: S3 key prefix (folder name)
    :param local_path: Local directory path to save the downloaded folder
    :return: True if folder was downloaded, else False
    """
    # Create an S3 client
    s3 = boto3.client('s3')

    try:
        # List objects in the bucket with the given prefix
        objects = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)['Contents']
        
        # Create local directory if it doesn't exist
        if not os.path.exists(local_path):
            os.makedirs(local_path)

        # Download each object
        for obj in objects:
            s3_key = obj['Key']
            file_path = os.path.join(local_path, s3_key[len(prefix):])
            s3.download_file(bucket_name, s3_key, file_path)
    except Exception as e:
        print(f"Error downloading folder: {e}")
        return False
    return True

def upload_file_to_s3(file_path, bucket_name, object_name):
    """
    Upload a file to an S3 bucket
    
    :param file_path: Path to the file to upload
    :param bucket_name: Name of the S3 bucket
    :param object_name: S3 object name (the key)
    :return: True if file was uploaded, else False
    """
    # Create an S3 client
    s3 = boto3.client('s3')
    
    try:
        # Upload the file
        s3.upload_file(file_path, bucket_name, object_name)
    except Exception as e:
        print(f"Error uploading file: {e}")
        return False
    return True

def download_file_from_s3(bucket_name, object_name, file_path):
    """
    Download a file from an S3 bucket
    
    :param bucket_name: Name of the S3 bucket
    :param object_name: S3 object name (the key)
    :param file_path: Path to save the downloaded file
    :return: True if file was downloaded, else False
    """
    # Create an S3 client
    s3 = boto3.client('s3')
    
    try:
        # Download the file
        s3.download_file(bucket_name, object_name, file_path)
    except Exception as e:
        print(f"Error downloading file: {e}")
        return False
    return True
