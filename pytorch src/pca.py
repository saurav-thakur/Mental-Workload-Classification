from sklearn.decomposition import PCA
import numpy as np
import constants
import torch

def extract_feature_using_pca(data):
    
    print("Feature Extraction Using PCA Started")
    data = torch.from_numpy(data)
    data = data.permute(2,0,1)
    data = np.array(data)


    final_data = []
    for i in range(len(data)):
        pca = PCA(n_components=62)
        final_data.append(pca.fit_transform(data[i]))
        
    final_data = np.array(final_data)
    final_data = final_data.reshape(360,1,62,62)
    print("Feature Extraction using PCA Completed")

    np.save(f"{constants.DATA_FOLDER_PATH}/pca_62x62_data.npy",final_data)
