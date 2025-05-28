 #🚢 Titanic Survivor Prediction

This project is a machine learning application that predicts whether a passenger on the Titanic would survive, based on features like age, class, fare, and more. It includes data processing, model training, and a Streamlit web app for interactive predictions.

---

#📌 Project Features

- 🧹 **Data Preprocessing**  
  Handle missing values, encode categorical variables, and scale features.

- 📊 **Exploratory Data Analysis (EDA)**  
  Visualize trends using barplots and boxplots to understand the dataset.

- 🧠 **Machine Learning Models**  
  - Logistic Regression  
  - Random Forest Classifier

- 🌐 **Streamlit Web App**  
  A user-friendly interface where you can input passenger details and get survival predictions instantly.

# 🗂️ File Structure

Titanic-Survivor-Prediction/
│
├── eda.py # Exploratory Data Analysis
├── preprocessing.py # Data cleaning and feature engineering
├── feature_engg.py # Feature extraction and transformation
├── model_logistic.py # Logistic Regression model
├── model_randforest.py # Random Forest model
├── load_data.py # Data loading utility
├── streamlit_ui.py # Streamlit app for live prediction
├── test.py # Script for testing pipeline
├── titanic.zip # Dataset (CSV inside)
├── .gitignore # Git ignore file

# ▶️ Run Locally

#### Clone the repository

bash
git clone https://github.com/your-username/Titanic-Survivor-Prediction.git
cd Titanic-Survivor-Prediction

📈 Sample Visualization
Survival rate by passenger class:

![Screenshot 2025-05-28 200925](https://github.com/user-attachments/assets/4562eba1-93a6-4812-8ba8-b9a8518e433b)

🛠️ Built With
Python

pandas, numpy

scikit-learn

matplotlib, seaborn

Streamlit

