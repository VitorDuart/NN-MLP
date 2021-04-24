import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch import nn
import numpy as np


class CarEvaluationDataset(Dataset):
    def __init__(self, str_path):
        # Loading CarDataset
        print('Loading Car Dataset ... ')
        self.ds_car = pd.read_csv(str_path)
        self.processed_ds_car = self.preprocess()
        self.processed_ds_car.evaluation = self.processed_ds_car.evaluation.astype('long')
        self.processed_ds_car = self.processed_ds_car.values

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
                i +=1
        return new_ds_car
    
    def __len__(self):
        return len(self.processed_ds_car)

    def __getitem__(self, idx):
        sample = self.processed_ds_car[idx]
        X, y = sample[:-1], sample[-1]
        return torch.tensor(X.tolist()), y


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer_one = nn.Linear(6,5)
        self.layer_two = nn.Linear(5,5)
        self.layer_three = nn.Linear(5,4)
    
    def forward(self, X):
        output = self.layer_one(X)
        output = self.layer_two(torch.relu(output))
        output = self.layer_three(torch.relu(output))
        output = torch.softmax(output)
        return output

car_dataset = CarEvaluationDataset('datasets/car.csv')
train_dataset, test_dataset = random_split(
    car_dataset, 
    [int(len(car_dataset)*0.2), len(car_dataset)-int(len(car_dataset)*0.2)]
)


train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True)



#train_dataloader = DataLoader.(())


# device = "cuda" if torch.cuda.is_available() else "cpu"
# print("Using {} device".format(device))


# model =  Net().to(device)

# print(model)