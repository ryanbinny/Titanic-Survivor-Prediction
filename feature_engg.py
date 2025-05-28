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
