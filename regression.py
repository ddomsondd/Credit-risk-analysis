import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, cross_val_predict
from data_preparation import data_preparation_one_hot

def plot_reg(y_test, y_pred, target_column):
    plt.scatter(y_test, y_pred, color='purple')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='green', linestyle='--', label='Perfect fit')

    plt.xlabel('y_test')
    plt.ylabel('y_pred')
    plt.title(f'Plot for Linear Regression model for {target_column} column')
    plt.legend()
    plt.grid()

    #plt.show()
    plt.savefig(f'plots/plot_reg_{target_column}.png', bbox_inches='tight')


def create_regression_model(y_test, y_pred, target_column):
    r2 = r2_score(y_test, y_pred)
    mse = round(np.sqrt(mean_squared_error(y_test, y_pred)), 3)
    mae = round(mean_absolute_error(y_test, y_pred), 3)

    plot_reg(y_test, y_pred, target_column)
    return r2, mse, mae


def fill_zero_values(data, target_column, model):
    train_data = data[data[target_column] != 0]
    test_data = data[data[target_column] == 0]

    X_train = train_data.drop([target_column], axis=1)
    y_train = train_data[target_column]

    cross_val = cross_val_score(model, X_train, y_train, cv=5, scoring='r2') 
    y_pred = cross_val_predict(model, X_train, y_train, cv=5)

    r2 = r2_score(y_train, y_pred)
    mse = np.sqrt(mean_squared_error(y_train, y_pred))
    mae = mean_absolute_error(y_train, y_pred)

    model.fit(X_train, y_train)

    X_test = test_data.drop([target_column], axis=1)
    predictions = model.predict(X_test)
    data.loc[data[target_column] == 0, target_column] = predictions

    results = {
        'R2 Score:': r2,
        'MSE': mse,
        'MAE': mae
    }
    return data


def data_modification():
    one_hot_data = data_preparation_one_hot('regression')

    model_income = LinearRegression()
    one_hot_data = fill_zero_values(one_hot_data, 'income', model_income)

    model_assets = LinearRegression()
    one_hot_data = fill_zero_values(one_hot_data, 'assets_value', model_assets)
    
    """
    df_income = pd.DataFrame(results_income)
    print(df_income)

    df_assets = pd.DataFrame(results_assets)
    print(df_assets)
    """
    
    return one_hot_data


    