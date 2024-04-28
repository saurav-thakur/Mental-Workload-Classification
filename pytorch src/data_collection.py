import os
import cv2
import numpy as np
import pandas as pd
from zipfile import is_zipfile,ZipFile
import constants
import scipy.io
from constants import PREPROCESSED, DATA_FOLDER_PATH
from fft import extract_feature_using_fft
from pca import extract_feature_using_pca

def load_and_preprocess_data(data):

    print("----------------------------------------")
    print(f"Collecting {PREPROCESSED} Data")
    print("----------------------------------------")

    if PREPROCESSED == "pca":
        try:
            final_data = np.load(f"{DATA_FOLDER_PATH}/pca_62x62_data.npy")
            final_data = final_data.reshape(360,1,62,62)

        except FileNotFoundError:
            extract_feature_using_pca(data)
            final_data = np.load(f"{DATA_FOLDER_PATH}/pca_62x62_data.npy")

        except Exception as e:
            raise e


        return final_data
    
    else:
        try:
            #spelling mistake
            final_data = np.load(f"{DATA_FOLDER_PATH}/fft_feature_selected_62.npy")
            

        except FileNotFoundError:
            extract_feature_using_fft(data)
            final_data = np.load(f"{DATA_FOLDER_PATH}/fft_feature_selected_62.npy")
        
        except Exception as e:
            raise e
        
        return final_data

            


def collect_data(path,dataset_type):
    
    mat_data = scipy.io.loadmat(path)
    label = mat_data["label"]
    data = mat_data["data"]
    label = np.squeeze(label)
    data = load_and_preprocess_data(data)

    return data,label



