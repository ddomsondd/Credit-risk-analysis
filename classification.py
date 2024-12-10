import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from imblearn.metrics import specificity_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
from data_preparation import data_preparation_one_hot
from regression import data_modification
import seaborn as sns
import statsmodels.graphics.gofplots as sm


def plot_confmat(y_test, y_pred, title):
    cm = confusion_matrix(y_test, y_pred)

    cm_disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    cm_disp.plot(cmap = plt.cm.Purples)
    plt.title(f"Confusion matrix for {title} model")
    plt.savefig(f'plots/models/plot_class_{title}_en.png', bbox_inches='tight')

    cm_disp.plot(cmap = plt.cm.Purples)
    plt.title(f"Macierz pomyłek dla modelu {title}")
    plt.xlabel("Wartość przewidziana")
    plt.ylabel("Wartość prawdziwa")
    plt.savefig(f'plots/models/plot_class_{title}_pl.png', bbox_inches='tight')


def prob_plots(data, title):
    for column in data.columns:
        plt.figure(figsize=(8, 6))
        sm.qqplot(data[column], line='s', alpha=0.5)
        plt.title(f"Q-Q Plot for {column} ({title})")
        plt.savefig(f'plots/probplots/qqplot_{column}_{title}.png', bbox_inches='tight')
        plt.close()

def create_classification_model(model, X, y, model_name):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = round(accuracy_score(y_test, y_pred), 4)
    recall = round(recall_score(y_test, y_pred), 4)
    specificity = round(specificity_score(y_test, y_pred), 4)
    f1 = round(f1_score(y_test, y_pred), 4)
    kappa = round(cohen_kappa_score(y_test, y_pred), 4)
    
    if model_name == 'Random Forest Classifier':
        headers = X.columns.to_list()

        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(10, 6))
        plt.title("Feature Importances")
        plt.bar(range(X.shape[1]), importances[indices], color='purple')
        plt.xticks(range(X.shape[1]), X, rotation=90)
        plt.xlabel("Importance")
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Features")
        plt.savefig(f'plots/more_infos/features_importance_rfc_en.png', bbox_inches='tight')

        plt.figure(figsize=(10, 6))
        plt.title("Ważność cech")
        plt.bar(range(X.shape[1]), importances[indices], color='purple')
        plt.xticks(range(X.shape[1]), X, rotation=90)
        plt.xlabel("Ważność")
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Cecha")
        plt.savefig(f'plots/more_infos/features_importance_rfc_pl.png', bbox_inches='tight')

    
    plot_confmat(y_test, y_pred, model_name)

    return accuracy, recall, specificity, f1, kappa


def execute_classification(one_hot_data):
    X = one_hot_data.drop('credit_risk', axis=1)
    y = one_hot_data['credit_risk']

    #checking samples distribution
    count_class = y.value_counts()
    #print(count_class)
    #english version
    plt.bar(count_class.index, count_class.values, color='purple')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('Class Distribution')
    plt.xticks(count_class.index, ['Class 0', 'Class 1'])
    plt.savefig(f'plots/more_infos/sample_distribution_en.png', bbox_inches='tight')

    #polish version
    plt.bar(count_class.index, count_class.values, color='purple')
    plt.xlabel('Klasa')
    plt.ylabel('Liczba próbek')
    plt.title('Rozkład klas')
    plt.xticks(count_class.index, ['Klasa 0', 'Klasa 1'])
    plt.savefig(f'plots/more_infos/sample_distribution_pl.png', bbox_inches='tight')


    #class balancing
    smote=SMOTE(sampling_strategy='minority') 
    X,y=smote.fit_resample(X,y)
    #print(y.value_counts())

    count_class = y.value_counts()
    #print(count_class)
    #english version
    plt.bar(count_class.index, count_class.values, color='pink')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('Class distribution after using SMOTE methode')
    plt.xticks(count_class.index, ['Class 0', 'Class 1'])
    plt.savefig(f'plots/more_infos/sample_distribution_smote_en.png', bbox_inches='tight')

    #polish version
    plt.bar(count_class.index, count_class.values, color='pink')
    plt.xlabel('Klasa')
    plt.ylabel('Liczba próbek')
    plt.title('Rozkład klas po użyciu metody SMOTE')
    plt.xticks(count_class.index, ['Klasa 0', 'Klasa 1'])
    plt.savefig(f'plots/more_infos/sample_distribution_smote_pl.png', bbox_inches='tight')


    models = [DecisionTreeClassifier(), RandomForestClassifier(), SVC(), GaussianNB(), XGBClassifier()]
    models_names = ['Decision Tree Classifier', 'Random Forest Classifier', 'Support Vector Classifier', 'Gausian Naive Bayes', 'XGBoost Classifier']
    results = []

    for model, model_name in zip(models, models_names):
        accuracy, recall, specificity, f1, kappa = create_classification_model(model, X, y, model_name)
        results.append({
            'Model': model_name,
            'Accuracy': accuracy,
            'Recall': recall,
            'Specificity': specificity,
            'F1 Score': f1,
            'Cohen Kappa': kappa,
        })

    return results

#WITH MEAN
#one_hot_data = data_preparation_one_hot('mean')

#prob_plots(one_hot_data.drop(columns=['credit_risk', 'credit_history', 'overdue_payments', 'employment_type', 'owns_property', 'education', 'city', 'marital_status'], axis=1), title="One-Hot Data")
#WITH REGRESSION
#one_hot_data = data_modification()

#WITH KNN
#one_hot_data = data_preparation_one_hot('knn')


#execute_classification(one_hot_data)