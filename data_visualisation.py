import pandas as pd
from data_preparation import data_preparation, change_zero_values_to_mean, change_zero_values_with_knn
import matplotlib.pyplot as plt
import seaborn as sns
from regression import data_modification

#WITH MEAN
data = data_preparation()
data = change_zero_values_to_mean()
#data = change_zero_values_with_knn()

#AGE vs. INCOME PLOT
#polish bersion
mean_income_by_age = data.groupby('age')['income'].mean().reset_index()
plt.figure(figsize=(8,6))
plt.plot(mean_income_by_age['age'], mean_income_by_age['income'], color='purple', label='Średni roczny przychód', linewidth=2)
plt.title('Średni roczny przychód w zależności od wieku')
plt.xlabel('Wiek')
plt.ylabel('Roczny przychód (zł)')
plt.legend(title='Legenda')
#plt.show()
plt.savefig(f'plots/visualisations/age_income_pl.png')

#english version
plt.figure(figsize=(8,6))
plt.plot(mean_income_by_age['age'], mean_income_by_age['income'], color='purple', label='Średni roczny przychód', linewidth=2)
plt.title('Average annual income depending on age')
plt.xlabel('Age')
plt.ylabel('Annual income (zł)')
plt.legend(title='Legend', labels=['Average annual income'])
#plt.show()
plt.savefig(f'plots/visualisations/age_income_en.png')


# INCOME DISTRIBUTION PLOT
#polish version
plt.figure(figsize=(8, 6))
sns.histplot(data['income'], bins=10, kde=True, color='#f6bcba')
plt.title('Rozkład rocznych przychodów')
plt.xlabel('Roczny przychód (zł)')
plt.ylabel('Liczba ludzi')
plt.savefig(f'plots/visualisations/income_pl.png')

#english version
plt.figure(figsize=(8, 6))
sns.histplot(data['income'], bins=10, kde=True, color='#f6bcba')
plt.title('Distribution of annual income')
plt.xlabel('Annual income (zł)')
plt.ylabel('Number of people')
plt.savefig(f'plots/visualisations/income_en.png')


#EMPLOYMENT TYPE vs. CREDIT RISK PLOT
#polish version
plt.figure(figsize=(8, 6))
sns.countplot(data=data, x='employment_type', hue='credit_risk', palette='rocket')
plt.title('Typ zatrudnienia a ryzyko kredytowe')
plt.xlabel('Typ zatrudnienia')
plt.ylabel('Liczba osób')
plt.legend(title='Ryzyko kredytowe', labels=['Niskie', 'Wysokie'])
plt.savefig(f'plots/visualisations/employment_type_credit_risk_pl.png')

#english version
plt.figure(figsize=(8, 6))
sns.countplot(data=data, x='employment_type', hue='credit_risk', palette='rocket')
plt.title('Employment type to credit risk')
plt.xlabel('Employment type')
plt.ylabel('Number of people')
plt.legend(title='Credit risk', labels=['Low', 'High'])
plt.savefig(f'plots/visualisations/employment_type_credit_risk_en.png')


# CHILDREN vs. CREDIT HISTORY PLOT
credit_history_by_children = (
    data.groupby('children')['credit_history']
    .value_counts(normalize=True)
    .unstack()
    .reset_index()
)

#polish version
plt.figure(figsize=(10, 6))
sns.countplot(data=data, x='children', hue='credit_history', palette='rocket')
plt.title('Liczba osób z różną historią kredytową w podziale na liczbę dzieci')
plt.xlabel('Liczba dzieci')
plt.ylabel('Liczba osób')
plt.legend(title='Historia kredytowa', labels=['Brak historii', 'Dobra historia', 'Zła historia'])
plt.savefig(f'plots/visualisations/children_credit_history_pl.png')

#english version
plt.figure(figsize=(10, 6))
sns.countplot(data=data, x='children', hue='credit_history', palette='rocket')
plt.title('Number of people with different credit histories divided by the number of children')
plt.xlabel('Number of children')
plt.ylabel('Number of people')
plt.legend(title='Credit history', labels=['No history', 'Good history', 'Bad history'])
plt.savefig(f'plots/visualisations/children_credit_history_en.png')


#CORRELATION MATRIX
data = data.drop(columns=['credit_history', 'overdue_payments', 'employment_type', 'owns_property', 'education', 'city', 'marital_status'], axis=1)
df = pd.DataFrame(data)
corr = df.corr()
#polish version
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".3f", cmap="Purples", linewidths=0.5, linecolor="black")
plt.title("Macierz korelacji")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(f'plots/visualisations/corr_matrix_pl.png', bbox_inches='tight')

#english version
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".3f", cmap="Purples", linewidths=0.5, linecolor="black")
plt.title("Correlation Matrix")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(f'plots/visualisations/corr_matrix_en.png', bbox_inches='tight')



"""
#WITH REGRESSION
data = data_modification()
#AGE vs. INCOME PLOT
mean_income_by_age = data.groupby('age')['income'].mean().reset_index()
plt.figure(figsize=(8,6))
plt.plot(mean_income_by_age['age'], mean_income_by_age['income'], color='purple', label='Średni roczny przychód', linewidth=2)
plt.title('Średni roczny przychód w zależności od wieku')
plt.xlabel('Wiek')
plt.ylabel('Roczny przychód (zł)')
plt.legend(title='Legenda')
plt.savefig(f'plots/visualisations/age_income_regression.png')

# INCOME DISTRIBUTION PLOT
plt.figure(figsize=(8, 6))
sns.histplot(data['income'], bins=10, kde=True, color='#f6bcba')
plt.title('Rozkład rocznych przychodów')
plt.xlabel('Roczny przychód (zł)')
plt.ylabel('Liczba ludzi')
plt.savefig(f'plots/visualisations/income_regression.png')
"""