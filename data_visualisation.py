import pandas as pd
from data_preparation import data_preparation
import matplotlib.pyplot as plt
import seaborn as sns

data = data_preparation()

#AGE vs. INCOME PLOT
mean_income_by_age = data.groupby('age')['income'].mean().reset_index()
plt.figure(figsize=(8,6))
sns.scatterplot(data=data, x='age', y='income', hue='education', palette='rocket')
plt.plot(mean_income_by_age['age'], mean_income_by_age['income'], color='purple', label='średni roczny przychód', linewidth=2)
plt.title('Age vs. Annual income')
plt.xlabel('Age')
plt.ylabel('Annual income (zł)')
plt.legend(title='Legend')
plt.savefig(f'plots/age_income.png')

# INCOME DISTRIBUTION PLOT
plt.figure(figsize=(8, 6))
sns.histplot(data['income'], bins=10, kde=True, color='#f6bcba')
plt.title('Annual income distribution')
plt.xlabel('Annual income (zł)')
plt.ylabel('Number of people')
plt.savefig(f'plots/income.png')

# CHILDREN vs. CREDIT HISTORY PLOT
plt.figure(figsize=(8, 6))
sns.countplot(data=data, x='children', hue='credit_history', palette='rocket')
plt.title('Number of children vs. Credit history')
plt.xlabel('Number of children')
plt.ylabel('Number of people')
plt.legend(title='Historia kredytowa')
plt.savefig(f'plots/children_credit_history.png')

# EMPLOYMENT TYPE PLOT 
plt.figure(figsize=(8, 6))
sns.countplot(data=data, x='employment_type', hue='education', palette='rocket')
plt.title('Employment type in relation to education')
plt.xlabel('Employment type')
plt.ylabel('Number of people')
plt.legend(title='Wykształcenie')
plt.savefig(f'plots/employment_type.png')



















