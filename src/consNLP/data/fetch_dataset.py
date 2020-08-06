# -*- coding: utf-8 -*-
import argparse
import os
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from tqdm import tqdm
#import zipfile
import shutil

def fetch_dataset(project_dir,download_from_kaggle=False, kaggle_dataset=None, kaggle_competition=None, download_from_s3=False, s3_bucket=None, download_from_url=False, data_url=None):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    logger.info('project directory {}'.format(project_dir))

    output_path = os.path.join(project_dir, 'data', 'raw')

    try:
        os.makedirs(output_path)
    except OSError:
        pass

    logger.info('output path {}'.format(output_path))

    if download_from_kaggle:
        os.environ['KAGGLE_USERNAME'] = os.environ['username']
        os.environ['KAGGLE_KEY'] = os.environ['kaggle_key']

        from kaggle.api.kaggle_api_extended import KaggleApi
        kaggle_api = KaggleApi()
        kaggle_api.authenticate()

        if kaggle_dataset:
            kaggle_api.dataset_download_files(kaggle_dataset,output_path,unzip=True)

        if kaggle_competition:
            kaggle_api.competition_download_files(kaggle_competition,path=output_path)

    if download_from_s3:
        import boto3
        session = boto3.Session(
                    aws_access_key_id=os.environ['aws_access_key_id'],
                    aws_secret_access_key=os.environ['aws_secret_access_key']
                    )
        s3 = session.resource('s3')

        if s3_bucket:
            my_bucket = s3.Bucket(s3_bucket)
            # download file into current directory
            for s3_object in tqdm(my_bucket.objects.all()):
                filename = s3_object.key
                logger.info("downloading {}".format(filename))
                my_bucket.download_file(s3_object.key, os.path.join(output_path,filename))

                try:
                    #zf = zipfile.ZipFile(os.path.join(output_path,filename), 'r')
                    #zf.extractall(output_path)
                    #zf.close()
                    shutil.unpack_archive(os.path.join(output_path,filename),output_path)

                    os.remove(os.path.join(output_path,filename))
                except shutil.ReadError:
                    pass

    if download_from_url and data_url:
        import urllib3

        http = urllib3.PoolManager()
        filename = data_url.split('/')[-1]
        logger.info("downloading {}".format(filename))

        with open(os.path.join(output_path,filename), 'wb') as out:
            r = http.request('GET', data_url, preload_content=False)
            shutil.copyfileobj(r, out)

        try:
            #zf = zipfile.ZipFile(os.path.join(output_path, filename), 'r')
            #zf.extractall(output_path)
            #zf.close()

            shutil.unpack_archive(os.path.join(output_path, filename), output_path)

            os.remove(os.path.join(output_path, filename))

        except shutil.ReadError:
            pass

    logger.info('download complete')
