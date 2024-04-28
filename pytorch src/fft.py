import numpy as np
import math
import scipy.io
from torch import nn,fft
import torch
import matplotlib.pyplot as plt
import constants

# extract important feature using FFT


def extract_feature_using_fft(data):
    
    print("Feature Extraction Using FFT Started")

    # converting data to torch tensor
    data = torch.from_numpy(data)
    data = data.permute(2,0,1)
    data = data.view(-1,62,2,256)

    # taking fft along dimension 3 which is 256
    fft_result = fft.fft(data, dim=3)
    fft_magnitude = torch.abs(fft_result)

    # extarcting 62 important features
    fft_result_selected = fft_magnitude[:, :, :, :62]

    # reshaping the data
    fft_result_selected = fft_result_selected.permute(0,2,1,3)

    # converting data to numpy
    fft_result_selected = np.array(fft_result_selected)
    # saving the data
    np.save(f"{constants.DATA_FOLDER_PATH}/fft_feature_selected_62.npy",fft_result_selected)
