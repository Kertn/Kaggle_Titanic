import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as F


pd.set_option('display.max_columns', None)
train_data = pd.read_csv('train.csv')

print(train_data.head())

print("/////////////////")

train_data = train_data.drop(columns=['Name', 'Ticket', 'Cabin', 'Embarked', 'PassengerId'])

train_data.loc[train_data['Sex'] == 'female', 'Sex'] = 2
train_data.loc[train_data['Sex'] == 'male', 'Sex'] = 1

train_data['Pclass'] = train_data['Pclass'].fillna(0)
train_data['Age'] = train_data['Age'].fillna(0)
train_data['SibSp'] = train_data['SibSp'].fillna(0)
train_data['Parch'] = train_data['Parch'].fillna(0)
train_data['Fare'] = train_data['Fare'].fillna(0)

y = train_data['Survived'].to_numpy()

train_data = train_data.drop(columns=['Survived'])

X = train_data.to_numpy()

print('y', y)
print('X', X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

print('X_train', X_train)
print('X_test', X_test)


X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

print(y_train)