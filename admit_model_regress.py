import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
import pickle



df=pd.read_csv('Admission_Predict.csv') 

df['Chance of admit class'] = df['Chance of Admit '].apply(lambda x:1 if x > 0.80 else 0)
# print(df.head())

columns = ['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA', 'Research']
X=df[columns]
y=df.iloc[:,9]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)
linReg = LinearRegression()
linReg.fit(X_train, y_train)

y_pred = linReg.predict(X_test)
print(linReg.predict([[330, 115, 1, 4.4, 4.8, 8.55, 1]]))





