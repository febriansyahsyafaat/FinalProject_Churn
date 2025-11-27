import pandas as pd
import joblib

def create_input_from_form(credit_score, gender, age, tenure, balance, 
                           num_products, has_card, is_active, salary, geography):
    input_dict = {
        'CreditScore': credit_score,
        'Gender': 1 if gender == 'Male' else 0,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_products,
        'HasCrCard': has_card,
        'IsActiveMember': is_active,
        'EstimatedSalary': salary,
        'Geography_Germany': 1 if geography == 'Germany' else 0,
        'Geography_Spain': 1 if geography == 'Spain' else 0
    }
    return input_dict
