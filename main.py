import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.optim as optim
from dataset import MNISTData
from classifier import ANNModel, CNNModel
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt


def init():
    train_data = pd.read_csv(r"./data/train.csv", dtype=np.float32)
    test_kag_data = pd.read_csv(r"./data/test.csv", dtype=np.float32)
    y = train_data.label.to_numpy()
    X = train_data.iloc[:, 1:].to_numpy() / 255
    X_kag = test_kag_data.to_numpy() / 255
    m, n = X.shape
    print(X.shape, y.shape, X_kag.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
    X_train_torch, X_test_torch = torch.from_numpy(X_train), torch.from_numpy(X_test)
    y_train_torch, y_test_torch = torch.from_numpy(y_train).type(torch.LongTensor), torch.from_numpy(y_test).type(torch.LongTensor)
    X_kag_torch = torch.from_numpy(X_kag)
    train_dataset = TensorDataset(X_train_torch, y_train_torch)
    test_dataset = TensorDataset(X_test_torch, y_test_torch)
    submit_dataset = TensorDataset(X_kag_torch, torch.tensor([0 for i in range(28000)]))
    train_loader = DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=100, shuffle=True)
    kag_loader = DataLoader(dataset=submit_dataset, batch_size=100, shuffle=False)

    return m, n, X_train, y_train, X_test, y_test, kag_loader, train_loader, test_loader


def show_image(images, labels, index):
    plt.imshow(images[index].reshape(28, 28))
    plt.axis("off")
    plt.title(str(labels[index]))
    plt.savefig('graph.png')
    plt.show()


def plot_loss(losses, iterations):
    plt.plot(iterations, losses)
    plt.xlabel("Number of iteration")
    plt.ylabel("Loss")
    plt.title("Logistic Regression: Loss vs Number of iteration")
    plt.show()


def plot_accuracy(iterations, accuracy_list):
    plt.plot(iterations, accuracy_list, color="red")
    plt.title("Accuracy Function")
    plt.xlabel("iterations")
    plt.ylabel("Accuracy")
    plt.show()


data_size, n_features, X_train, y_train,  X_test, y_test, kag_loader, train_loader, test_loader = init()
#show_image(X_train, y_train, 10)
#model = ANNModel(n_features, 100, 10)
model = CNNModel()
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)


def submit_test():
    submit_example = pd.read_csv('./data/sample_submission.csv')
    predictions = []
    for x, y in kag_loader:
        y_pred = model(x.view(100, 1, 28, 28))
        predicted = torch.max(y_pred.data, 1)[1]
        predictions.append(predicted.numpy())
    predictions = np.array(predictions).reshape(28000, -1)
    preds = np.array([x[0] for x in predictions])
    print(preds.shape)
    result = {"ImageId": submit_example.ImageId, "Label": preds}
    pd.DataFrame(result, columns=["ImageId", "Label"]).to_csv("./sample_submission.csv", index=False)


def train(epochs):
    model.train()
    count, loss_arr, iterations, accuracy_list = 0, [], [], []
    for e in range(epochs):
        for image, label in train_loader:
            image = image.view(100, 1, 28, 28)
            out = model(image)
            loss = loss_func(out, label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            count += 1
            if count % 50 == 0:
                # Calculate Accuracy
                correct = 0
                total = 0
                # Predict test dataset
                for images, labels in test_loader:
                    test = Variable(images.view(100, 1, 28, 28))

                    # Forward propagation
                    outputs = model(test)

                    # Get predictions from the maximum value
                    predicted = torch.max(outputs.data, 1)[1]

                    # Total number of labels
                    total += len(labels)

                    # Total correct predictions
                    correct += (predicted == labels).sum()

                accuracy = 100 * correct / float(total)

                # store loss and iteration
                loss_arr.append(loss.data)
                iterations.append(count)
                accuracy_list.append(accuracy)
            if count % 500 == 0:
                # Print Loss
                print('Iteration: {}  Loss: {}  Accuracy: {}%'.format(count, loss.data, accuracy))
    #plot_loss(loss_arr, iterations)
    #plot_accuracy(iterations, accuracy_list)
    submit_test()


train(10)



