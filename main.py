import numpy as np
import pandas as pd
from Network import Neurons



pd.set_option('display.max_columns', None)
train_data = pd.read_csv('train.csv')



del train_data[train_data.columns[3]]
del train_data[train_data.columns[0]]
del train_data[train_data.columns[6]]
del train_data[train_data.columns[8]]
train_data.loc[train_data['Sex'] == 'female', 'Sex'] = 2
train_data.loc[train_data['Sex'] == 'male', 'Sex'] = 1
train_data.loc[train_data['Cabin'] == 'T', 'Cabin'] = 0
train_data['Cabin'] = train_data['Cabin'].fillna(0)
train_data['Pclass'] = train_data['Pclass'].fillna(0)
train_data['Age'] = train_data['Age'].fillna(0)
train_data['SibSp'] = train_data['SibSp'].fillna(0)
train_data['Parch'] = train_data['Parch'].fillna(0)
train_data['Fare'] = train_data['Fare'].fillna(0)
train_data['Fare'] = train_data['Fare']/10
train_data['Age'] = train_data['Age']/10

answers = train_data['Survived']
del train_data[train_data.columns[0]]

values = {'A' : '7.', 'B' : '6.', 'C' : '5.', 'D' : '4.', 'E' : '3.', 'F' : '2.', 'G' : '1.'}

training_data = []
for data, answ in zip(train_data.values, answers):
    if data[-1] != 0:
        if len(data[-1].split(" ")) > 1:
            data[-1] = data[-1].split(" ")[0]
        data[-1] = round(float(data[-1].replace(f'{data[-1][0]}', values[data[-1][0]])))
    training_data.append((data, answ))
print(np.shape(training_data[0][0]))
test_data = training_data[:50]
training_data = training_data[100:150]

training_data[0] = (np.array([-100, -100, -100, -100, -100, -100, -1000], dtype=object), 0)



"""
Titanic = Neurons([7, 4, 2, 1])
Titanic.SGD(training_data, 100, 1, 0.01, test_data=test_data)
Titanic.make_graph()
"""