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

train.shape
test.shape