import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_train = pd.read_csv('loan_train.csv')
data_test = pd.read_csv('loan_test.csv')
X_train = data_train.iloc[:,1:12].values
y_train = data_train.iloc[:,12].values
X_test = data_test.iloc[:,1:12].values



