import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler    
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import json

# Loading Data
df = pd.read_excel('Combineeed_data.xlsx', sheet_name='Append2')

# Cleaning
df=df.dropna(subset=['Name','University'])
df.isnull().sum()
df.loc[df['Gre_1'] <= 10, 'Gre_1'] = None
df.loc[df['Gre_1'] <= 150, 'Gre_1'] = None
df.loc[df['Gre_2'] <=150, 'Gre_2'] = None
df.loc[df['Tofel_1'] >=150, 'Tofel_1'] = None
df.loc[df['Tofel_2'] >=150, 'Tofel_2'] = None


df.loc[df['Tofel_1'] <=10,'Tofel_1'] = None

df.loc[df['Tofel_2'] <=10,'Tofel_2'] = None

df.loc[df['ILETS_1'] >=11,'ILETS_1'] = None

df.loc[df['ILETS_2'] >=11,'ILETS_2'] = None

df['Gre_1'] = df['Gre_1'].fillna(df['Gre_1'].mean())

df['Gre_2'] = df['Gre_1'].fillna(df['Gre_2'].mean())

df['Tofel_1'] = df['Tofel_1'].fillna(df['Tofel_1'].mean())

df['Tofel_2'] = df['Tofel_2'].fillna(df['Tofel_2'].mean())

df['ILETS_1'] = df['ILETS_1'].fillna(df['ILETS_1'].mean())

df['ILETS_2'] = df['ILETS_2'].fillna(df['ILETS_2'].mean())

df['GRE'] = df['Gre_1'].combine_first(df['Gre_2'])

df['TOFEL'] = df['Tofel_1'].combine_first(df['Tofel_2'])

df['ILETS'] = df['ILETS_1'].combine_first(df['ILETS_2'])

df['Status'] = df['Status'].replace({'Yes': 1, 'No': 0})

# Preparing dataset for training
In_e = pd.get_dummies(df['Intake'])

df=df.dropna(subset=['University'])


le = LabelEncoder()
## Encode column A
df['A_encoded'] = le.fit_transform(df['University'])

df = df.drop(['Name','University','Intake','Year','Masters','Gre_1','Gre_2','Tofel_1', 'Tofel_2','English_Test','English_Test_1','Test_Score','Test_Score_2','ILETS_1','ILETS_2'], axis=1)

encoded_df = pd.concat([df,In_e], axis=1)
encoded_df=encoded_df.dropna(subset=['Status'])

x = encoded_df.iloc[:,1:].values  
y = encoded_df['Status']
 
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.25, random_state=0)
  
## feature Scaling  
st_x= StandardScaler()    
x_train= st_x.fit_transform(x_train)    
x_test= st_x.transform(x_test)    

# Training
xgb_classifier = xgb.XGBClassifier()
xgb_classifier.fit(x_train,y_train)
y_pred = xgb_classifier.predict(x_test)

# Accuracy
acc=accuracy_score(y_test,y_pred)
acc=acc*100
print(acc)

# Finding precision and recall
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy   :", accuracy)
precision = precision_score(y_test, y_pred)
print("Precision :", precision)
recall = recall_score(y_test, y_pred)
print("Recall    :", recall)
F1_score = f1_score(y_test, y_pred)
print("F1-score  :", F1_score)

filename = open("predict_xgboost.pkl", "wb")
pickle.dump(xgb_classifier, filename)