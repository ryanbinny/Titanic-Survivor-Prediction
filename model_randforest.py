import pandas as pd
df=pd.read_csv(r"E:\Programming\Titanic Survivor Prediction\titanic\train.csv")

def clean_data(df):
    df['Age']=df['Age'].fillna(df['Age'].median())
    df['Embarked']=df['Embarked'].fillna(df['Embarked'].mode()[0])
    df.drop(["Cabin","Ticket","Name"],axis=1,inplace=True)
    df=pd.get_dummies(df,columns=['Sex','Embarked'],drop_first=True)
    return df
df=clean_data(df)

def feature_splits(df):
    x=df.drop(['Survived','PassengerId'],axis=1)
    y=df['Survived']
    return x,y
x,y=feature_splits(df)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

model=RandomForestClassifier(class_weight='balanced',random_state=42)
model.fit(x_train_scaled,y_train)

#Accuracy reports
y_pred=model.predict(x_test_scaled)
print("Accuracy:",accuracy_score(y_test,y_pred))
print("Confusion Matrix:",confusion_matrix(y_test,y_pred))
print("Classification Report:",classification_report(y_test,y_pred))

import joblib
joblib.dump(model, "E:/Programming/Titanic Survivor Prediction/titanic_model.pkl")
joblib.dump(scaler, "E:/Programming/Titanic Survivor Prediction/scaler.pkl")
