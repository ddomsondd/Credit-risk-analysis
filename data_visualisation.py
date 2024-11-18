import pandas as pd
from data_preparation import data_preparation
import matplotlib.pyplot as plt
import seaborn as sns

data = data_preparation()
data = pd.DataFrame(data)
data_short = data.head()
data_short = str(data_short.to_html())

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


data = pd.read_csv('data_atlas.csv')

df = pd.DataFrame(data)
df_short = df.head()
table_view = str(df_short.to_html())



script = '''
<!DOCTYPE html>
<html>
<head>    
    <title>Atlas Bootcamp</title>
    <link rel="stylesheet" href="style.css">
</head>
<body>
  <div class="container">

    <div class="header">
      <h1>Credit risk analysis</h1>
    </div>

    <div class="main">
      <p>I started working on the report by familiarizing myself with the data set and analyzing the data.</p>
''' + table_view + '''
      <p style="font-style: italic; font-size: 14px; text-align: center; margin-top: 2px;">Table 1. A representation of the first few rows of the dataset.</p>
      <p>
        Some columns, like 'Unnamed: 0' and 'support_indicator,' do not contain useful information and can be removed. 
        Some columns need to be transformed to make the data easier to interpret, such as removing unnecessary strings like "dzieci" or "złoty". 
        Rows with missing values (NaN) will be deleted, but to avoid losing too much data, some NaN values will be replaced with random values from the same column. 
        Columns will also be converted into one-hot encoded columns. All data processing functions are included in the <span style="font-style: italic;">data_preparation.py</span> file. 
        The cleaned data is then ready for preliminary analysis.
      </p>
      ''' + data_short + '''
      <p style="font-style: italic; font-size: 14px; text-align: center; margin-top: 2px;">Table 2. A representation of the first few rows of a dataset after data processing.</p>
      
      <h3>Sample charts for the data set</h3>
      <div class="img_class">
        <div class="box">
          <p style="width: 80%;">
            <img src="plots/age_income.png" arc="" style="float: left; margin-right: 30px;" /> 
            A chart comparing age to annual salary, taking into account the type of education. Since there are many samples, simply including all the data does not provide much information. 
            Only places of denser and rarer dots representing samples are visible. That's why I added a solid chart that shows the average annual salary by specific age group. 
            So you can see that it fluctuates mainly in the range of PLN 20-25 thousand per year.
          </p> 
        </div>

        <div class="box">
          <p style="width: 80%;">
            <img src="plots/children_credit_history.png" arc="" style="float: right; margin-left: 30px;">
            The chart shows the relationship between the number of children and credit history. It illustrates that most loans are taken by individuals without children. 
            With each additional child, the number of loans decreases.
          </p>
        </div>

        <div class="box">
          <p style="width: 80%;">
            <img src="plots/income.png" arc="" style="float: left; margin-right: 30px;" >
            The chart shows the distribution of average annual income in relation to the number of people. The largest group earns around 40,000 PLN annually, followed by incomes in the range of 15,000–20,000 PLN per year.
          </p>
        </div>

        <div class="box">
          <p style="width: 80%;">
            <img src="plots/employment_type.png" arc="" style="float: right; margin-left: 30px;">
            The chart shows employment types in relation to education level. An interesting observation is that, for each employment type, 
            the majority of individuals have a secondary education, followed by primary education, rather than higher education as one might expect.          </p>
        </div>
      </div>

      <h3>Classifier models</h3>
      <div>
        <p>I compared five classification models:
          <ul>
            <li>Decision Tree Classifier</li>
            <li>Random Forest Classifier</li>
            <li>Support Vector Classifier</li>
            <li>Gaussian Naive Bayes</li> 
            <li>XGBoost Classifier</li>
          </ul>
          As you can see from the table below, the best-fitting models are random forest and XGBoost. The accuracy and specificity statistics of all models are very similar and high, but the exception is the sensitivity index, which in the case of the support vector machine and the naive Bayes classifier indicates 0, which may indicate some errors or problems with the data.
        </p>
        <div>
          <table class="dataframe">
            <thead>
              <tr style="text-align: right;"> 
                <th></th>
                <th>Model</th>
                <th>Accuracy</th>
                <th>Recall</th>
                <th>Specificity</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <th>0</th>
                <td>DTC</td>
                <td>0.9865</td>
                <td>0.7727</td>
                <td>0.9906</td>
              </tr>
              <tr>
                <th>1</th>
                <td>RFC</td>
                <td>0.9941</td>
                <td>0.6818</td>
                <td>1.0000</td>
              </tr>
              <tr>
                <th>2</th>
                <td>SVC</td>
                <td>0.9815</td>
                <td>0.0000</td>
                <td>1.0000</td>
              </tr>
              <tr>
                <th>3</th>
                <td>GNB</td>
                <td>0.9815</td>
                <td>0.0000</td>
                <td>1.0000</td>
              </tr>
              <tr>
                <th>4</th>
                <td>XGBC</td>
                <td>0.9941</td>
                <td>0.6818</td>
                <td>1.0000</td>
              </tr>
            </tbody>
          </table>
          <p style="font-style: italic; font-size: 14px; text-align: center; margin-top: 2px;">Table 3. Presentation of the performance metrics for the model fitting.</p>

        </div>
        
        <h4>Confusion matrix for each model</h4>
        <div class="img_class">
          <img src="plots/plot_class_DTC.png" arc="Confusion matrix for Decision Tree Classifier">
          <img src="plots/plot_class_RFC.png" arc="Confusion matrix for Random Forest Classifier">
          <img src="plots/plot_class_SVC.png" arc="Confusion matrix for Support Machine Classifier">
          <img src="plots/plot_class_GNB.png" arc="Confusion matrix for Gaussian Naive Bayes Classifier">
          <img src="plots/plot_class_XGBC.png" arc="Confusion matrix for XGBoost Classifier">
        </div>
        <p>The confusion matrices indicate high accuracy in classifying examples, but it is evident that the dataset has been significantly reduced. 
          Initially, the dataset contained around 10,000 rows, so the test set should have about 2,000 samples. However, the confusion matrices show only 1,189 samples. 
          This reduction is due to the removal of rows with missing values (NaN). This is not necessarily a bad number (probably ;P), because initially, 
          I only used the method of deleting rows with NaN values, which resulted in only 585 samples, meaning the dataset was reduced by almost 3.5 times. 
          After applying the method of replacing some NaN values with random values from the respective columns, the dataset was much less reduced.</p>

      </div>
    </div>
    <div class="footer"></div>
  </div>
</body>
</html> 
'''

'''
f = open('report_english.html', 'w', encoding='utf-8')
f.write(script)
f.close()
'''




#wszystkie dane
#dane przerobione
#jakieś wykresiki proste
#cos do czegos
#jakas regresja cos do czegos??
#one hot
#klasyfikacja
#pierwszy taki raport 