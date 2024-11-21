import pandas as pd
from random import choice

def data_check():
    data = pd.read_csv('data_atlas.csv')
    headers = data.columns.to_list()
    print(headers)

    #checking columns names and unique values 
    df = pd.DataFrame(data)
    
    for column in df.columns:
        print(f"Liczba wystąpień wartości w kolumnie '{column}':")
        print(df[column].value_counts())
        print()
    
    print(df['overdue_payments'].value_counts())
    print(df['owns_property'].value_counts())
    print(df['credit_history'].value_counts())
    print(df['credit_history'].unique())


    print(df['overdue_payments'].unique())
    print(df['employment_type'].unique())
    print(df['owns_property'].unique())
    print(df['education'].unique())
    print(df['city'].unique())
    print(df['marital_status'].unique())
    
def data_preparation():
    data = pd.read_csv('data_atlas.csv')
    
    data['credit_history'] = data['credit_history'].fillna('zła historia')
    data['overdue_payments'] = data['overdue_payments'].fillna(choice(['brak opóźnień', 'opóźnienia']))
    data['owns_property'] = data['owns_property'].fillna(choice(['tak', 'nie']))

    data = data.dropna()
    data = data.drop(['Unnamed: 0', 'support_indicator '], axis=1)

    data = data[data['overdue_payments'].isin(['brak opóźnień', 'opóźnienia'])]
    data['children'] = data['children'].str.replace(' dzieci', '').replace('brak', '0').astype(int)
    data['income'] = data['income'].str.replace(' złoty', '').astype(int)
    data['assets_value'] = data['assets_value'].str.replace(' złoty', '').astype(int)

    return data

def data_preparation_one_hot():
    data = data_preparation()

    one_hot_data = pd.get_dummies(data, columns = ['credit_history', 'overdue_payments', 'employment_type', 'owns_property', 'education', 'city', 'marital_status'], dtype=int)
    #one_hot_data.info()
    return one_hot_data
