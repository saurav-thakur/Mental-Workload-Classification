import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import constants


def train_model(model,data, loss_fn, optimizer):

  model.train()
  train_loss, train_acc = 0, 0

  for batch in data:
      X, y = batch

      y_pred = model(X)

      loss = loss_fn(y_pred, y)
      train_loss += loss.item() 

      optimizer.zero_grad()
      loss.backward()

      optimizer.step()
      y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
      train_acc += (y_pred_class == y).sum().item()/len(y_pred)

  train_loss = train_loss / len(data)
  train_acc = train_acc / len(data)
  return train_loss, train_acc


def test_model(model, data, loss_fn,device) :

    model.eval()
    test_loss, test_acc = 0, 0

    with torch.inference_mode():
        for batch, (X, y) in enumerate(data):
            X, y = X.to(device), y.to(device)

            test_prediction = model(X)

            loss = loss_fn(test_prediction, y)
            test_loss += loss.item()

            test_pred_labels = test_prediction.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))

    # getting avg loss and acc
    test_loss = test_loss / len(data)
    test_acc = test_acc / len(data)

    return test_loss, test_acc



def train(model,train_data, test_data, optimizer, loss_fn, epochs, device,  plots: bool,k_split):
  
    results = {"train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }

    for epoch in range(epochs):
        train_loss, train_acc = train_model(model=model,data=train_data,loss_fn=loss_fn, optimizer=optimizer)

        test_loss, test_acc = test_model(model=model, data=test_data,loss_fn=loss_fn,device=device)

        
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_acc: {test_acc:.4f} | "
        )

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    if plots:
        os.makedirs(f"model_plots/{constants.PREPROCESSED}",exist_ok=True)
        # Plot and save train results
        plt.figure()  # Create a new figure
        plt.plot(np.arange(0,epochs,1), results["train_loss"], label="Train Loss")
        plt.plot(np.arange(0,epochs,1), results["train_acc"], label="Train Accuracy")
        plt.title(f"Train Loss V/s Train Accuracy for {k_split} split")
        plt.xlabel("Epochs")
        plt.ylabel("Metrics")
        plt.legend()
        plt.savefig(f"model_plots/{constants.PREPROCESSED}/train_{k_split}.png")

        # Plot and save validation results
        plt.figure()  # Create a new figure
        plt.plot(np.arange(0,epochs,1), results["test_loss"], label="Validation Loss")
        plt.plot(np.arange(0,epochs,1), results["test_acc"], label="Validation Accuracy")
        plt.title(f"Validation Loss V/s Validation Accuracy for {k_split} split")
        plt.xlabel("Epochs")
        plt.ylabel("Metrics")
        plt.legend()
        plt.savefig(f"model_plots/{constants.PREPROCESSED}/validation_{k_split}.png")

        plt.figure()  # Create a new figure
        plt.plot(np.arange(0,epochs,1), results["train_acc"], label="Train Accuracy")
        plt.plot(np.arange(0,epochs,1), results["test_acc"], label="Validation Accuracy")
        plt.title(f"Train Accuracy V/s Validation Accuracy for {k_split} split")
        plt.xlabel("Epochs")
        plt.ylabel("Metrics")
        plt.legend()
        plt.savefig(f"model_plots/{constants.PREPROCESSED}/train_validation_acc_{k_split}.png")

    return results

