import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle


df = pd.read_csv('Admission_Predict.csv')

columns = ['GRE Score', 'TOEFL Score', 'University Rating', 'CGPA', 'Research']

X = df[columns]
y = df.iloc[:, 8]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


reg = LinearRegression()
reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)
# print(reg.predict([[330, 115, 1, 4.4, 4.8, 8.55, 1]]))

# open a file, where you ant to store the data
file = open('chances_regression.pkl', 'wb')

# dump information to that file
pickle.dump(reg, file)
