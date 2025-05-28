import os
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'titanic_model.pkl')
print("Model path:", model_path)
print("Exists:", os.path.exists(model_path))

model = joblib.load(model_path)
