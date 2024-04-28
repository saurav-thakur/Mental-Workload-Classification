import os
import numpy as np
import torch
from torch import nn
from data_setup import prepare_and_split_data,CustomDataset
from torch.utils.data import DataLoader
import constants
from model_builder import CNNModel
from torchsummary import summary
from torchvision import models
import engine
import utils
from sklearn.model_selection import KFold


kfold = KFold(n_splits=5,random_state=41,shuffle=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

data, labels = prepare_and_split_data(dataset_path=constants.DATASET_PATH,dataset_type="train")

train_acc = []
test_acc = []

print(f"{constants.PREPROCESSED} preprocessed data used!! ")

for idx,(train_idx, val_idx) in enumerate(kfold.split(data)):
    train_data, train_labels = data[train_idx], labels[train_idx]
    val_data, val_labels = data[val_idx], labels[val_idx]

    # converting data into train and validation data loader
    train_dataset = CustomDataset(data=train_data.float(), labels=train_labels, dataset_type="train")
    val_dataset = CustomDataset(data=val_data.float(), labels=val_labels, dataset_type="train")

    # Create DataLoader objects
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=64, shuffle=True)

    # model
    model = CNNModel(
        input_shape=constants.IMAGE_CHANNELS, output_shape=constants.NUMBER_OF_CLASS, hidden_layers=constants.HIDDEN_LAYERS
    )
    print(f"------------------------ {idx + 1} split ------------------------------------------")

    if constants.PREPROCESSED == "pca":
        summary(model, input_size=(1,62,62))
    else:
        summary(model, input_size=(2,62,62))
    


    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=constants.LEARNING_RATE)

    results = engine.train(model=model, train_data=train_dataloader,
                        test_data=val_dataloader, loss_fn=loss_fn,
                        optimizer=optimizer, epochs=constants.NUM_EPOCHS, device=device, plots=constants.PLOTS,k_split=idx)

    train_acc.append(results["train_acc"][-1])
    test_acc.append(results["test_acc"][-1]) 

print("Final Mean Train Accuracy is ", np.mean(train_acc))
print("Final Mean Test Accuracy is ", np.mean(test_acc))

if constants.SAVE_MODEL:
    utils.save_model(
        model=model, target_dir="trained_models",model_name="model_001.pt"
    )



