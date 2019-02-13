import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train = pd.read_csv('titanic_train.csv')
test = pd.read_csv('titanic_test.csv')
data = pd.concat([train, test])

#refining
train.drop(['Cabin'], axis=1, inplace=True)
test.drop(['Cabin'], axis=1, inplace=True)

#encoding gender
replace_gen = {'male':1, 'female':0}
train.Sex = train.Sex.replace(replace_gen)
test.Sex = test.Sex.replace(replace_gen)


train.Age = train.Age.transform(lambda x: x.fillna(x.median()))
test.Age = test.Age.transform(lambda x: x.fillna(x.median()))
test.Fare = test.Fare.transform(lambda x: x.fillna(x.median()))
test.apply(lambda x: sum(x.isnull()))
train.Embarked = train.Embarked.transform(lambda x: x.fillna('S'))

y_train = train.Survived
x_train = train.drop(['Survived'], axis=1, inplace=True)

replace_emb = {'C':0, 'Q':1, 'S':2}
train.Embarked = train.Embarked.replace(replace_emb)
test.Embarked = test.Embarked.replace(replace_emb)

X_train = train
X_test = test

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y = logreg.predict(X_train)


from sklearn import metrics
print(metrics.accuracy_score(y_train, y))

from sklearn import tree
model = tree.DecisionTreeClassifier(criterion='gini')

model.fit(X_train, y_train)
y_pred = model.predict(X_train)
print(metrics.accuracy_score(y_train, y_pred))
y_new = model.predict(X_test)













