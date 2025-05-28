import pandas as pd
df=pd.read_csv(r"E:\Programming\Titanic Survivor Prediction\titanic\train.csv")
import seaborn as sns
import matplotlib.pyplot as plt

def basic_info(df):
    print(df.info())
    print(df.describe())
    print(df.isnull().sum())

def plot_distributions(df):
    sns.countplot(x="Survived",data=df)
    plt.show()
    sns.countplot(x="Pclass",hue="Survived",data=df)
    plt.show()

basic_info(df)
plot_distributions(df)
