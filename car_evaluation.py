import pandas as pd
from sklearn.model_selection import train_test_split
import torch


# Loading CarDataset
print('Loading Car Dataset')
ds_car = pd.read_csv('datasets/car.csv')


# Preprocessing Car Dataset
print('Preprocessing dataset')

#Converting the categorical values to numeric 
for column in ds_car.columns:
    attributes_values = ds_car[column].drop_duplicates()
    i = 0.
    for attribute_value in attributes_values:
        ds_car.loc[ds_car[column]==attribute_value, column] = i
        i +=1

#Split car dataset with holdout validation
print('Spliting car dataset with holdout validation ...')
train_X, test_X, train_y, test_y = train_test_split(
    ds_car[ds_car.columns[0:5]],
    ds_car.evaluation,
    test_size = 0.2
)

train_X = torch.tensor(train_X.values.tolist()).long()
train_y = torch.tensor(train_y.values.tolist()).long()

test_X = torch.tensor(test_X.values.tolist()).long()
test_y = torch.tensor(test_y.values.tolist()).long()


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))


