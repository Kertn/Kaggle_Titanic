import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


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

answers = train_data['Survived'].values
del train_data[train_data.columns[0]]
del train_data[train_data.columns[-1]]
feature_matrix = train_data[train_data.columns].values

print(feature_matrix)
print(answers)

train_feature_matrix, test_feature_matrix, train_labels, test_labels = train_test_split(
    feature_matrix, answers, test_size=0.2, random_state=42)


BOB = KNeighborsClassifier()
params = {
    'n_neighbors': np.arange(1, 20),
    'metric': ['manhattan', 'euclidean'],
    'weights': ['uniform', 'distance']
}
BOB_grid = GridSearchCV(BOB, params, cv=5, scoring='accuracy', n_jobs=-1)
BOB_grid.fit(train_feature_matrix, train_labels)
print(BOB_grid.best_params_)
print(accuracy_score(test_labels, BOB_grid.best_estimator_.predict(test_feature_matrix)))