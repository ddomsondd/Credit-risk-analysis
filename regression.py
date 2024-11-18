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
from data_preparation import data_preparation_one_hot

def plot_reg(y_test, y_pred, title):
    plt.scatter(y_test, y_pred, color='purple')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='green', linestyle='--', label='Perfect fit')

    plt.xlabel('y_test')
    plt.ylabel('y_pred')
    plt.title(f'Plot of observed and fitted values for {title} model')
    plt.legend()
    plt.grid()

    #plt.show()
    plt.savefig(f'plots/plot_reg_{title}.png', bbox_inches='tight')


def create_regression_model(model, X, y, model_name):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mse = round(np.sqrt(mean_squared_error(y_test, y_pred)), 3)
    mae = round(mean_absolute_error(y_test, y_pred), 3)

    plot_reg(y_test, y_pred, model_name)
    return r2, mse, mae



one_hot_data = data_preparation_one_hot()

X = one_hot_data.drop('children', axis=1)
y = one_hot_data['children']

regression_models = [LinearRegression(), XGBRegressor()]
models_names = ['LR', 'XGBR']
results = []
for model, model_name in zip(regression_models, models_names):
    r2, mse, mae = create_regression_model(model, X, y, model_name)
    results.append({
        'Model': model_name,
        'R2 Score:': r2,
        'MSE': mse,
        'MAE': mae
    })

df = pd.DataFrame(results)
print(df)