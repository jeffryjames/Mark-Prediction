import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

data = pd.read_csv("D:/Python/test_scores.csv")
print(data.head())
df = pd.DataFrame(data)
print(df)

print(data.columns)

df.columns[df.isna().any()]

df = df.drop('school',axis=1)
df = df.drop('classroom',axis=1)
df = df.drop('n_student',axis=1)
df = df.drop('student_id',axis=1)
df = df.drop('lunch',axis=1)
df = df.drop('school_type',axis=1)

print(df)
print(df.info())

# Mapping String into Numbers
df['school_setting'] = df['school_setting'].map({'Urban':1,'Suburban':2,'Rural':3}).astype(int)
df['teaching_method'] = df['teaching_method'].map({'Standard':1,'Experimental':2}).astype(int)
df['gender'] = df['gender'].map({'Female':1,'Male':2}).astype(int)
print(df.head(5))

X = df.iloc[:,:-1].values
print(X)

y = df.iloc[:,-1].values
print(y)


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.25,random_state=0)

model = LinearRegression()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)
print(y_pred)

pickle.dump(model, open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))

print(model.predict([[1, 2, 1,100]]))



