import pandas as pd
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
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from data_preparation import data_preparation_one_hot


def plot_confmat(y_test, y_pred, title):
    cm = confusion_matrix(y_test, y_pred)

    cm_disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    cm_disp.plot(cmap = plt.cm.Purples)
    plt.title(f"Confusion matrix for {title} model")
    #plt.show()
    plt.savefig(f'plots/plot_class_{title}.png', bbox_inches='tight')


def create_classification_model(model, X, y, model_name):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = round(accuracy_score(y_test, y_pred), 4)
    recall = round(recall_score(y_test, y_pred), 4)
    specificity = round(specificity_score(y_test, y_pred), 4)

    plot_confmat(y_test, y_pred, model_name)

    return accuracy, recall, specificity


def execute_classification():
    one_hot_data = data_preparation_one_hot()

    X = one_hot_data.drop('credit_risk', axis=1)
    y = one_hot_data['credit_risk']


    models = [DecisionTreeClassifier(), RandomForestClassifier(), SVC(), GaussianNB(), XGBClassifier()]
    models_names = ['DTC', 'RFC', 'SVC', 'GNB', 'XGBC']
    results = []

    for model, model_name in zip(models, models_names):
        accuracy, recall, specificity = create_classification_model(model, X, y, model_name)
        results.append({
            'Model': model_name,
            'Accuracy': accuracy,
            'Recall': recall,
            'Specificity': specificity
        })

    return results

