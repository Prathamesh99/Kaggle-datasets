import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train = pd.read_csv('loan_train.csv')
test = pd.read_csv('loan_test.csv')
data = pd.concat([train, test])

replace_dict = {1.0:'Yes', 0.0:'No'}
train.Credit_History = train.Credit_History.replace(replace_dict)
test.Credit_History = test.Credit_History.replace(replace_dict)
train.Credit_History = train.Credit_History.fillna('Unknown')
test.Credit_History = test.Credit_History.fillna('Unknown')

replace_dict = {'0':'0', '1':'1', '2':'2', '3+':'3'}

train.Dependents = train.Dependents.replace(replace_dict)
test.Dependents = test.Dependents.replace(replace_dict)
train.Dependents = train.Dependents.fillna('Unknown')
test.Dependents = test.Dependents.fillna('Unknown')

train.LoanAmount = train.LoanAmount.transform(lambda x: x.fillna(x.mean()))
test.LoanAmount = test.LoanAmount.transform(lambda x: x.fillna(x.mean()))
train.Loan_Amount_Term = train.Loan_Amount_Term.transform(lambda x: x.fillna(x.mean()))
test.Loan_Amount_Term = test.Loan_Amount_Term.transform(lambda x: x.fillna(x.mean()))

train.Self_Employed = train.Self_Employed.fillna('Unknown')
test.Self_Employed = test.Self_Employed.fillna('Unknown')

train.Gender = train.Gender.fillna('Unknown')
test.Gender = test.Gender.fillna('Unknown')
train.Married = train.Married.fillna('Unknown')
test.Married = test.Married.fillna('Unknown')

data = pd.concat([train, test])
data.apply(lambda x: sum(x.isnull()))

import keras 
from keras.layers import Dense
