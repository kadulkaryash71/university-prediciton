import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle



df=pd.read_csv('combine.csv') 

# df['Chance of admit class'] = df['Chance of Admit '].apply(lambda x:1 if x > 0.80 else 0)
# print(df.head())

columns = ['GRE', 'English Test', 'English Score', 'University', 'CGPA', 'Work Experience']
X=df[columns]
y=df.iloc[:,4]

print(X , y)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)
classifier=RandomForestClassifier()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
# print(classifier.predict([[330, 115, 2, 4.4, 4.8, 8.55, 1]]))



