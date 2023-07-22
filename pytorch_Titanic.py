import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn


pd.set_option('display.max_columns', None)
train_data = pd.read_csv('train.csv')


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


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))


data_size = X_train.size()[0]
features_num = X_train.size()[1]



model = nn.Linear(features_num, 1)

criterion = nn.MSELoss()

learning_rate = 0.01

optimizer = torch.optim.SGD(model.parameters(), learning_rate)

epochs = 100

y_train = y_train.unsqueeze(1)

loss_list = []

for epoch in range(epochs):
    pred_model = model(X_train)
    loss_model = criterion(pred_model, y_train)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data)
    print("loss_model.detach().flatten()[0]", loss_model.detach().flatten()[0])
    loss_list.append(loss_model.detach().flatten()[0])
    loss_model.backward()
    optimizer.step()
    optimizer.zero_grad()

plt.figure(figsize=(12, 8))
plt.plot(range(len(loss_list)), loss_list)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.grid()
plt.show()

with torch.no_grad():
    pred_test = model(X_test)
    print(roc_auc_score(y_test, pred_test))

with torch.no_grad():
    y_predicted = model(X_test)
    y_predicted_cls = y_predicted.round()
    acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])
    print(f'accuracy = {acc:.4f}')