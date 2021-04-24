#!/usr/bin/env python
"""
File_Manager_Tar_Gz.py: 
download_file_tar_gz: Checks if download file already exists, otherwise downloads .tar.gz file from dataset link.
open_file_tar_gz: Checks if extracted folder already exists, otherwise extracts .tar.gz file.
unpickle: Read data from batch files. Function taken from https://www.cs.toronto.edu/~kriz/cifar.html
"""

__author__      = "Umer Fakher"

import tarfile
import os

import requests


def download_file_tar_gz(dataset_link = None, dataset_file_name = None):
    """ Checks if download file already exists, otherwise downloads .tar.gz file from dataset link. 
    
    :param dataset_link: Link for dataset
    :param dataset_file_name: String file name of .tar.gz file
    :return: True if performed function, False if file/folder already exists
    """


    if dataset_link is None:
        dataset_link = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    
    if dataset_file_name is None:
        dataset_file_name = "cifar-10-python.tar.gz"
    
    if not os.path.isfile(dataset_file_name):

        with open(dataset_file_name, "wb") as f:
            r = requests.get(dataset_link)
            f.write(r.content)
        print("File download as {}".format(dataset_file_name))
        return True

    else:
        print("Error: File already downloaded as file {}/".format(dataset_file_name))
        return False

def open_file_tar_gz(file_name, check_extracted_folder_name='cifar-10-batches-py'):
    """ Checks if extracted folder already exists, otherwise extracts .tar.gz file. 

    :param file_name: String file name of .tar.gz file
    :return: True if performed function, False if file/folder already exists
    """
    
    if not os.path.exists(check_extracted_folder_name):
        t = tarfile.open(file_name, "r:gz")
        t.extractall()
        t.close()
        print("File extracted as folder {}/".format(check_extracted_folder_name))
        return True
    else:
        print("Error: File already extracted as folder {}/".format(check_extracted_folder_name))
        return False

def unpickle(file_name):
    """ Open batch files.
    Function taken from: https://www.cs.toronto.edu/~kriz/cifar.html
    """
    import pickle
    with open(file_name, 'rb') as fo:
        dico = pickle.load(fo, encoding='bytes')
    return dico

if __name__ == "__main__":
    file_name = "cifar-10-python.tar.gz"
    website_link = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    
    print(download_file_tar_gz(dataset_link=website_link, dataset_file_name=file_name))
    print(open_file_tar_gz(file_name))