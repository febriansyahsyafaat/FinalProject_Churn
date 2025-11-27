import joblib
import os
import pandas as pd

class ChurnPredictor:
    def __init__(self, model_path='models/best_model.pkl', 
                 scaler_path='models/scaler.pkl',
                 feature_names_path='models/feature_names.pkl'):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        if os.path.exists(feature_names_path):
            self.feature_names = joblib.load(feature_names_path)
        else:
            self.feature_names = [
                'CreditScore', 'Gender', 'Age', 'Tenure', 'Balance',
                'NumOfProducts', 'HasCrCard', 'IsActiveMember', 
                'EstimatedSalary', 'Geography_Germany', 'Geography_Spain'
            ]
    
    def preprocess(self, input_data):
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        else:
            df = input_data.copy()
        df = df[self.feature_names]
        numerical_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary']
        df[numerical_cols] = self.scaler.transform(df[numerical_cols])
        return df
    
    def predict(self, input_data):
        df_processed = self.preprocess(input_data)
        prediction = self.model.predict(df_processed)[0]
        probability = self.model.predict_proba(df_processed)[0, 1]
        return int(prediction), float(probability)
