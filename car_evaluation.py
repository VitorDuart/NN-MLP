import pandas as pd
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np


class CarEvaluationDataset:
    def __init__(self, str_path):
        # Loading CarDataset
        print('Loading Car Dataset ... ')
        self.ds_car = pd.read_csv(str_path)
        self.X, self.y = self.preprocess()
        
    def __str__(self):
        return f'{self.ds_car}'
    
    def __len__(self):
        return len(self.processed_ds_car)

    def __getitem__(self, idx):
        if type(idx) is int: return (self.X.iloc[idx].values, self.y.iloc[idx])
        else: return (self.X.iloc[idx,].values, self.y.iloc[idx,].values)

    def getvaluesXy(self):
        return (self.X.values, self.y.values)

    # Preprocessing Car Dataset
    def preprocess(self):
        print('Preprocessing dataset ...')

        new_ds_car = pd.DataFrame(data=self.ds_car)

        columns = new_ds_car.columns.tolist()
        for column in columns:
            values = new_ds_car[column].drop_duplicates()
            v = 0
            for value in values:
                if column != 'evaluation':
                    new_column = f'{column}-{value}' 
                    new_ds_car[new_column] = new_ds_car[column]
                    new_ds_car.loc[new_ds_car[new_column] != value, new_column] = 0
                    new_ds_car.loc[new_ds_car[new_column] == value, new_column] = 1
                else:
                    new_ds_car.loc[new_ds_car[column] == value,column] = v
                v+=1

            if column != 'evaluation': new_ds_car.pop(column)
            
        X = new_ds_car.iloc[:, :-1]
        y = new_ds_car.iloc[:,-1]
        return (X,y)

def holdout_validation(dataset):
    X, y = dataset.getvaluesXy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    return (X_train, X_test, y_train, y_test)    

def build_model(shape):
    model = Sequential([
        Dense(16, input_shape = [shape[1]], activation='relu'),
        Dense(16, activation='relu'),
        Dense(16, activation='relu'),
        Dense(4, activation='softmax')
    ])

    return model

def fit_model(model, X_train, y_train, X_test, y_test, epochs):
    model.compile(
        optimizer='adam', 
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    #learning rate = 0.0001
    model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test))


    

car_dataset = CarEvaluationDataset('datasets/car.csv')

X_train, X_test, y_train, y_test = holdout_validation(dataset=car_dataset)

print(X_train)


model = build_model(X_train.shape)

fit_model(model=model, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, epochs=500)


