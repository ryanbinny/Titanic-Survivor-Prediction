import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'titanic_model.pkl')
scaler_path = os.path.join(BASE_DIR, 'scaler.pkl')

print("Model path:", model_path)
print("Scaler path:", scaler_path)

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

st.title("Titanic Survivor Prediction")

df = pd.read_csv(r"E:\Programming\Titanic Survivor Prediction\titanic\train.csv")

st.write("### Survival Rate by Passenger Class")

# Create columns to position the plot
col1, col2 = st.columns([2, 1])  

with col2:  # Put the plot in the right column
    fig, ax = plt.subplots(figsize=(4, 3))  # Set figure size
    sns.barplot(x='Pclass', y='Survived', data=df, ax=ax)
    st.pyplot(fig)


pclass=st.selectbox("Passenger class(1=1st, 2=2nd, 3=3rd)",[1,2,3])
age=st.slider("Age",0,100,30)
sibsp=st.number_input("No. of Siblings/Spouses on board:",0,10,0)
parch=st.number_input("No. of Parents/Children on board:",0,10,0)
fare=st.number_input("Fare",0.0,600.0,32.0)
sex_male=st.selectbox("Sex",["Male","Female"])
embarked_Q=st.selectbox("Embarked(Q=Queensburg, S=Southampton, C=Cherbourg)",['Q','S','C'])

data={
    "Pclass":pclass,
    "Age":age,
    "SibSp":sibsp,
    "Parch":parch,
    "Fare":fare,
    "Sex_male": 1 if sex_male=="Male" else 0,
    "Embarked_Q": 1 if embarked_Q=="Q" else 0,
    "Embarked_S": 1 if embarked_Q=="S" else 0
}

input_df=pd.DataFrame([data])
input_scaled=scaler.transform(input_df)

if st.button("Predict"):
    prediction=model.predict(input_scaled)[0]
    result="Survived" if prediction==1 else "Did not survive"

    st.subheader(f"Prediction:{result}")

    if prediction==1:
        st.success("Prediction:Survived")
    else:
        st.error("Prediction:Did not survive")