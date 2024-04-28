import numpy as np
import os
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import warnings
from sklearn.model_selection import KFold

from helper import *
warnings.filterwarnings("ignore")



if __name__ == "__main__":
    # Inputs

    # initializing k fold for 5 folds
    kfold = KFold(n_splits=5)
    lr = 0.01
    epochs = 100
    test_acc = []

    # loading data
    X,y = load_data(data_path="../dataset/WLDataCW.mat")


    for idx,(train_idx,test_idx) in enumerate(kfold.split(X.T)):
        print(f"------------- {idx+1} Split Running -------------")

        # accuracy and loss list to calculate mean
        loss_list = []
        accuracy_list = []

        # splitting data on folds
        X_train,X_test = X.T[train_idx],X.T[test_idx]
        y_train,y_test = y[train_idx],y[test_idx]

        # Transposing X to the initial state
        X_train = X_train.T
        X_test = X_test.T

        # weight initialization
        weight = np.random.randn(62*512,1)
        bias = np.random.randn(1, 1)

        
        for i in range(epochs):
            
            # forward propagation
            forward_output = forward_propagation(X=X_train,weight=weight, bias=bias)

            # calcuating the forward prop loss and accuracy
            loss = binary_cross_entropy(y_true=y_train, y_pred=forward_output)
            acc = accuracy_score(y_train,np.squeeze(forward_output.astype(int)))
            print(f"Epoch -> {i+1} || Loss -> {round(loss,3)} || Accuracy -> {round(acc,3)}")
            loss_list.append(round(loss,3))
            accuracy_list.append(round(acc,3))

            # backward propagation
            dl_w, dl_b = backprop(X_train, y_train, y_pred=forward_output)
            
            # updating weights based on backward prop
            weight = weight -  lr * dl_w
            bias = bias - lr * dl_b


        print()
        print(f"Mean train loss is {round(np.mean(loss),3)}")
        print(f"Mean train accuracy is {round(np.mean(accuracy_list),3)}")
        print()

        # testing the weights produced by our model
        predicted_output = forward_propagation(X_test,weight,bias)
        predicted_output = predicted_output.astype(int)

        print()
        tst_acc = accuracy_score(y_test.reshape(-1,1),predicted_output.reshape(-1,1))
        print(f"Test Accuracy -> {round(tst_acc,3)}")
        test_acc.append(tst_acc)
        print()



print()
print(f"Final Mean Test Accuracy is {round(np.mean(test_acc),3)}\n")
