import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch import nn
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

class CarEvaluationDataset(Dataset):
    def __init__(self, str_path):
        # Loading CarDataset
        print('Loading Car Dataset ... ')
        self.ds_car = pd.read_csv(str_path)
        self.processed_ds_car = self.preprocess()
        self.processed_ds_car.evaluation = self.processed_ds_car.evaluation.astype('long')

    # Preprocessing Car Dataset
    def preprocess(self):
        print('Preprocessing dataset ...')

        new_ds_car = pd.DataFrame(data=self.ds_car)

        #Converting the categorical values to numeric 
        for column in new_ds_car.columns:
            attributes_values = new_ds_car[column].drop_duplicates()
            i = 0.
            for attribute_value in attributes_values:
                new_ds_car.loc[new_ds_car[column]==attribute_value, column] = i
                i +=1.
        return new_ds_car
    
    def __len__(self):
        return len(self.processed_ds_car)

    def __getitem__(self, idx):
        processed_ds_car = self.processed_ds_car.values
        sample = processed_ds_car[idx]
        X, y = sample[:-1], sample[-1]
        return torch.tensor(X.tolist()), y

class Net(nn.Module):
    # def __init__(self):
    #     super(Net, self).__init__()
    #     self.layer_one = nn.Linear(6,5)
    #     self.layer_two = nn.Linear(5,5)
    #     self.layer_three = nn.Linear(5,4)

    def __init__(self):
        super(Net, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Linear(64, 30),
            nn.ReLU(),
            nn.Linear(30, 30),
            nn.ReLU(),
            nn.Linear(30, 30),
            nn.Linear(30, 4),
            nn.ReLU(),
            nn.Softmax()
        )
    
    # def forward(self, X):
    #     print(X)
    #     output = self.layer_one(X)
    #     print(output)
    #     output = self.layer_two(torch.relu(output))
    #     print(output)
    #     output = self.layer_three(torch.relu(output))
    #     print(output)
    #     output = torch.softmax(output)
    #     print(output)
    #     return output

    def forward(self, X):
        return self.linear(X)

def holdout_validation(dataset):
    train_dataset, test_dataset = random_split(
        dataset, 
        [int(len(dataset)*0.2), len(dataset)-int(len(dataset)*0.2)]
    )
    
    return (train_dataset, test_dataset)

def fit_model(model, train_dataset, epochs, eta):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=eta)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    for epoch in range(epochs):
        acc = 0
        for X, y in train_dataloader:
            X, y = X.to(device), y.to(device)

            #forward
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation and Optimizer
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc += (pred.argmax(axis=1) == y).sum()
        print(f"epoch={epoch} loss={loss/len(train_dataset)} acc={acc/len(train_dataset)}")

car_dataset = CarEvaluationDataset('datasets/car.csv')

train_dataset, test_dataset = holdout_validation(dataset=car_dataset)

test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True)


model = Net().to(device)


fit_model(model=model, train_dataset=train_dataset, epochs=200, eta=0.03)
