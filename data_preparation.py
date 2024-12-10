import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
import seaborn as sns
import matplotlib.pyplot as plt

def data_check():
    data = pd.read_csv('data_atlas.csv')
    headers = data.columns.to_list()
    print(headers)

    #checking columns names and unique values 
    df = pd.DataFrame(data)
     
    for column in df.columns:
        print(f"Liczba wystąpień wartości w kolumnie '{column}':")
        print(df[column].value_counts())
        print(df[column].unique())
        print(df[column].isna().sum())
        print()

    
    
def data_preparation():
    data = pd.read_csv('data_atlas.csv')

    data = data.drop(['Unnamed: 0', 'support_indicator '], axis=1)

    data['credit_history'] = data['credit_history'].fillna('zła historia')
    data['overdue_payments'] = data['overdue_payments'].fillna('unknown')
    data['owns_property'] = data['owns_property'].fillna('unknown')
    data['children'] = data['children'].str.replace(' dzieci', '').replace('brak', '0').astype(int)
    data['income'] = data['income'].fillna('0 złoty')
    data['income'] = data['income'].str.replace(' złoty', '').astype(int)
    data['assets_value'] = data['assets_value'].fillna('0 złoty')
    data['assets_value'] = data['assets_value'].str.replace(' złoty', '').astype(int)

    return data


def change_zero_values_to_mean():
    data = data_preparation()
    mean_income = int(data[data['income'] != 0]['income'].mean())
    #print(mean_income)
    data.loc[data['income'] == 0, 'income'] = mean_income

    mean_assets_value = int(data[data['assets_value'] != 0]['assets_value'].mean())
    #print(mean_assets_value)
    data.loc[data['assets_value'] == 0, 'assets_value'] = mean_assets_value
    income_zero_count = data[data['income'] == 0].shape[0]
    assets_value_zero_count = data[data['assets_value'] == 0].shape[0]
    #print(f'INCOME 0: {income_zero_count}')
    #print(f'ASSETS_VALUE 0: {assets_value_zero_count}')
    
    return data


def change_zero_values_with_knn():
    data = data_preparation()
    imputer = KNNImputer(n_neighbors=5)
    data['income'] = data['income'].replace(0, np.nan)
    data['income'] = imputer.fit_transform(data[['income']])

    data['assets_value'] = data['assets_value'].replace(0, np.nan)
    data['assets_value'] = imputer.fit_transform(data[['assets_value']])
    
    return data


def data_preparation_one_hot(operation):
    if operation == 'mean':
        data = change_zero_values_to_mean()
    elif operation == 'knn':
        data = change_zero_values_with_knn()
    else:
        data = data_preparation()
    one_hot_data = pd.get_dummies(data, columns = ['credit_history', 'overdue_payments', 'employment_type', 'owns_property', 'education', 'city', 'marital_status'], dtype=int)

    return one_hot_data
