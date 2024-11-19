import pandas as pd
from data_preparation import data_preparation
from classification import execute_classification


data = pd.read_csv('data_atlas.csv')
data = pd.DataFrame(data)
table_view_1 = str(data.head().to_html())

data_2 = data_preparation()
data_2 = pd.DataFrame(data_2)
table_view_2 = str(data_2.head().to_html())

results_classification = execute_classification()
results_classification = pd.DataFrame(results_classification)
results_classification_table = str(results_classification.to_html())

script_english = '''
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
''' + table_view_1 + '''
      <p style="font-style: italic; font-size: 14px; text-align: center; margin-top: 2px;">Table 1. A representation of the first few rows of the dataset.</p>
      <p>
        Some columns, like 'Unnamed: 0' and 'support_indicator,' do not contain useful information and can be removed. 
        Some columns need to be transformed to make the data easier to interpret, such as removing unnecessary strings like "dzieci" or "złoty". 
        Rows with missing values (NaN) will be deleted, but to avoid losing too much data, some NaN values will be replaced with random values from the same column. 
        Columns will also be converted into one-hot encoded columns. All data processing functions are included in the <span style="font-style: italic;">data_preparation.py</span> file. 
        The cleaned data is then ready for preliminary analysis.
      </p>
      ''' + table_view_2 + '''
            <p style="font-style: italic; font-size: 14px; text-align: center; margin-top: 2px;">Table 2. A representation of the first few rows of a dataset after data processing.</p>
      
      <h2>Sample charts for the data set</h2>
      <p>All the charts I created in the file <span style="font-style: italic;">data_visualistion.py</span>.</p>
      <div class="img_class">
        <div class="box">
          <div style="flex-direction: column;">
            <img src="plots/age_income.png" arc="" style="margin-right: 30px;">
            <p style="font-style: italic; font-size: 14px; text-align: center; margin-top: 2px;">Chart 1.</p>
          </div>
            A chart comparing age to annual salary, taking into account the type of education. Since there are many samples, simply including all the data does not provide much information. 
            Only places of denser and rarer dots representing samples are visible. That's why I added a solid chart that shows the average annual salary by specific age group. 
            So you can see that it fluctuates mainly in the range of PLN 20-25 thousand per year.
        </div>

        <div class="box">
          <div style="flex-direction: column;">
            <img src="plots/children_credit_history.png" arc="" style="margin-right: 30px;">
            <p style="font-style: italic; font-size: 14px; text-align: center; margin-top: 2px;">Chart 2.</p>
          </div>
            The chart shows the relationship between the number of children and credit history. It illustrates that most loans are taken by individuals without children. 
            With each additional child, the number of loans decreases.
        </div>

        <div class="box">
          <div style="flex-direction: column;">
            <img src="plots/income.png" arc="" style="margin-right: 30px;" >
            <p style="font-style: italic; font-size: 14px; text-align: center; margin-top: 2px;">Chart 3.</p>
          </div>            
          The chart shows the distribution of average annual income in relation to the number of people. The largest group earns around 40,000 PLN annually, followed by incomes in the range of 15,000–20,000 PLN per year.
        </div>

        <div class="box">
          <div style="flex-direction: column;">
            <img src="plots/employment_type.png" arc="" style="margin-right: 30px;" >
            <p style="font-style: italic; font-size: 14px; text-align: center; margin-top: 2px;">Chart 4.</p>
          </div>
            The chart shows employment types in relation to education level. An interesting observation is that, for each employment type, 
            the majority of individuals have a secondary education, followed by primary education, rather than higher education as one might expect.
        </div>
      </div>

      <h2>Classifier models</h2>
      <p>I compared five classification models. I created the models in the file <span style="font-style: italic;">classification.py</span>.
        <ul>
          <li>Decision Tree Classifier</li>
          <li>Random Forest Classifier</li>
          <li>Support Vector Classifier</li>
          <li>Gaussian Naive Bayes</li> 
          <li>XGBoost Classifier</li>
        </ul>
        As you can see from the table below, the best-fitting models are random forest and XGBoost. The accuracy and specificity statistics of all models are very similar and high, but the exception is the sensitivity index, which in the case of the support vector machine and the naive Bayes classifier indicates 0, which may indicate some errors or problems with the data.
      </p>
      
    ''' + results_classification_table + '''
      <p style="font-style: italic; font-size: 14px; text-align: center; margin-top: 2px;">Table 3. Presentation of the performance metrics for the model fitting.</p>  
      
      <h3>Confusion matrix for each model</h3>
      <div class="img_inline">
        <div style="flex-direction: column;">
          <img src="plots/plot_class_DTC.png" arc="Confusion matrix for Decision Tree Classifier">
          <p style="font-style: italic; font-size: 14px; text-align: center; margin-top: 2px;">1. Confusion matrix for Decision Tree Classification model.</p>
        </div>
        <div style="flex-direction: column;">
          <img src="plots/plot_class_RFC.png" arc="Confusion matrix for Random Forest Classifier">
          <p style="font-style: italic; font-size: 14px; text-align: center; margin-top: 2px;">2. Confusion matrix for Random Forest Classification model.</p>
        </div>
        <div style="flex-direction: column;">
          <img src="plots/plot_class_SVC.png" arc="Confusion matrix for Support Vector Machine Classifier">
          <p style="font-style: italic; font-size: 14px; text-align: center; margin-top: 2px;">3. Confusion matrix for Support Vector Machine Classification model.</p>
        </div>
        <div style="flex-direction: column;">
          <img src="plots/plot_class_GNB.png" arc="Confusion matrix for Gaussian Naive Bayes Classifier">
          <p style="font-style: italic; font-size: 14px; text-align: center; margin-top: 2px;">4. Confusion matrix for Gaussian Naive Bayes model.</p>
        </div>
        <div style="flex-direction: column;">
          <img src="plots/plot_class_XGBC.png" arc="Confusion matrix for XGBoost Classifier">
          <p style="font-style: italic; font-size: 14px; text-align: center; margin-top: 2px;">5. Confusion matrix for XGBoost Classification model.</p>
        </div>
      </div>
      
      <p>The confusion matrices indicate high accuracy in classifying examples, but it is evident that the dataset has been significantly reduced. 
        Initially, the dataset contained around 10,000 rows, so the test set should have about 2,000 samples. However, the confusion matrices show only 1,189 samples. 
        This reduction is due to the removal of rows with missing values (NaN). This is not necessarily a bad number (probably ;P), because initially, 
        I only used the method of deleting rows with NaN values, which resulted in only 585 samples, meaning the dataset was reduced by almost 3.5 times. 
        After applying the method of replacing some NaN values with random values from the respective columns, the dataset was much less reduced.
      </p>
    </div>
    <div class="footer"></div>
  </div>
</body>
</html> 
'''



f = open('report_english.html', 'w', encoding='utf-8')
f.write(script_english)
f.close()



script_polish = '''
<!DOCTYPE html>
<html>
<head>    
    <title>Atlas Bootcamp</title>
    <link rel="stylesheet" href="style.css">
</head>
<body>
  <div class="container">

    <div class="header">
      <h1>Analiza zdolności kredytowej</h1>
    </div>

    <div class="main">
      <p>Pracę nad raportem zaczęłam od zapoznania się ze zbiorem danych oraz analizy danych.</p>
    ''' + table_view_1 + '''
      <p style="font-style: italic; font-size: 14px; text-align: center; margin-top: 2px;">Tabela 1. Przedstawienie kilku pierwszych wierszy zbioru danych.</p>
      <p>
          Niektóre kolumny, takie jak: 'Unnamed: 0', 'support_indicator ', nie wnoszą żadnych istotnych danych, więc można je od razu usunąć.
          Należy także przekształcić dane w niektórych kolumnach tak, aby były łatwo interpretowane przez program, np. poprzez usunięcie niepotrzebnych łańcuchów znakowych ("dzieci", "złotych"), usunięcie wierszy z wybrakowanymi danymi, tzn. wierszy, które zawierają komórki z wartościami NaN, a także przekształcenie kolumn na kolumny one-hot encoding. 
          Aby nie stracić jednak za dużo danych niektóre wartości NaN zostaną zastąpione przez losowo wybrane wartości występujące w danej kolumnie.
          Wszelkie funkcje do obróbki danych zawarłam w pliku <span style="font-style: italic;">data_preparation.py</span>.
          Tak przygotowane dane można poddać wstępnej analizie.
      </p>
    ''' + table_view_2 + '''
      <p style="font-style: italic; font-size: 14px; text-align: center; margin-top: 2px;">Tabela 2. Przedstawienie kilku pierwszych wierszy zbioru danych po przetworzeniu danych.</p>

      <h2>Przykładowe wykresy do zbioru danych</h2>
      <p>Wszystkie wykresy utworzyłam w pliku <span style="font-style: italic;">data_visualistion.py</span>.</p>
      <div class="img_class">
        <div class="box">
          <div style="flex-direction: column;">
            <img src="plots/age_income.png" arc="" style="margin-right: 30px;">
            <p style="font-style: italic; font-size: 14px; text-align: center; margin-top: 2px;">Wykres 1.</p>
          </div>
            Wykres zestawiający wiek do rocznego wynagrodzenia z uwzględnieniem rodzaju wykształcenia. Jako że próbek jest dużo, samo uwzględnienie wszystkich danych nie daje zbyt dużo informacji. Widać jedynie miejsca gęstszego oraz rzadszego wystąpienia kropek przedstawiających próbki. Dlatego też dodałam wykres ciągły, który przedstawia średnie wynagrodzenie roczne według określonej grupy wiekowej. Widać więc, że oscyluje ono głównie w przedziale 20-25 tysięcy zł rocznie.
        </div>

        <div class="box">
          <div style="flex-direction: column;">
            <img src="plots/children_credit_history.png" arc="" style="margin-right: 30px;">
            <p style="font-style: italic; font-size: 14px; text-align: center; margin-top: 2px;">Wykres 2.</p>
          </div>
            Wykres przedstawia zestawienie liczby dzieci do historii kredytowej. Jak widać najwięcej kredytów jest brane przez osoby nie posiadające dzieci. Z każdnym kolejnym dzieckiem liczba wziętych kredytów jest mniejsza.
        </div>

        <div class="box">
          <div style="flex-direction: column;">
            <img src="plots/income.png" arc="" style="margin-right: 30px;" >
            <p style="font-style: italic; font-size: 14px; text-align: center; margin-top: 2px;">Wykres 3.</p>
          </div>
            Wykres przedstawia rozkład średniego rocznego wynagrodzenia w odniesieniu do ilości osób. Jak widać największa liczba przebadanych zarabia około 40 tys. zł rocznie, zaś na drugim i trzecim miejscu są dochody z zakresu 15-20 tys. zł rocznie.
        </div>

        <div class="box">
          <div style="flex-direction: column;">
            <img src="plots/employment_type.png" arc="" style="margin-right: 30px;">
            <p style="font-style: italic; font-size: 14px; text-align: center; margin-top: 2px;">Wykres 4.</p>
          </div>
            Wykres przedstawia typy zatrudnienia w odniesieniu do rodzaju wykształcenia. Ciekawym spostrzeżeniem jest fakt, iż w każdnym z typów zatrudnień najwięcej przebadanych posiada wykształcenie średnie, następnie podstawowe, a nie jak mogłoby się wydawać - wyższe.
        </div>
      </div>

      <h2>Modele klasyfikacyjne</h2>
      <p>Porównałam ze sobą pięć modeli klasyfikacyjnych. Modele stworzyłam w pliku <span style="font-style: italic;">classification.py</span>.
        <ul>
          <li>Decision Tree Classifier</li>
          <li>Random Forest Classifier</li>
          <li>Support Vector Classifier</li>
          <li>Gaussian Naive Bayes</li> 
          <li>XGBoost Classifier</li>
        </ul>
        Jak widać po poniższej tabeli najlepiej dopasowane modele to las losowy oraz XGBoost. Statystyki dokładności oraz swoistości wszystkich modeli są do siebie bardzo zbliżone i wysokie, ale jednak wyjątek stanowi wskaźnik czułości, który w przypadku maszyny wektorów nośnych oraz naiwnego klasyfikatora Bayesa wskazuje 0, co może wskazywać na pewne błędy lub problemy z danymi. 
      </p>
    ''' + results_classification_table + '''
      <p style="font-style: italic; font-size: 14px; text-align: center; margin-top: 2px;">Tabela 2. Przedstawienie miar jakości dopasowania danych modeli.</p>
      
      <h3>Macierze pomyłek dla poszczególnych modeli</h3>
      <div class="img_inline">
        <div style="flex-direction: column;">
          <img src="plots/plot_class_DTC.png" arc="Confusion matrix for Decision Tree Classifier">
          <p style="font-style: italic; font-size: 14px; text-align: center; margin-top: 2px;">1. Macierz pomyłek dla modelu drzewa decyzyjnego.</p>
        </div>
        <div style="flex-direction: column;">
          <img src="plots/plot_class_RFC.png" arc="Confusion matrix for Random Forest Classifier">
          <p style="font-style: italic; font-size: 14px; text-align: center; margin-top: 2px;">2. Macierz pomyłek dla modelu lasu losowego.</p>
        </div>
        <div style="flex-direction: column;">
          <img src="plots/plot_class_SVC.png" arc="Confusion matrix for Support Vector Machine Classifier">
          <p style="font-style: italic; font-size: 14px; text-align: center; margin-top: 2px;">3. Macierz pomyłek dla modelu maszyny wektorów nośnych.</p>
        </div>
        <div style="flex-direction: column;">
          <img src="plots/plot_class_GNB.png" arc="Confusion matrix for Gaussian Naive Bayes Classifier">
          <p style="font-style: italic; font-size: 14px; text-align: center; margin-top: 2px;">4. Macierz pomyłek dla naiwnego klasyfikatora Bayesa.</p>
        </div>
        <div style="flex-direction: column;">
          <img src="plots/plot_class_XGBC.png" arc="Confusion matrix for XGBoost Classifier">
          <p style="font-style: italic; font-size: 14px; text-align: center; margin-top: 2px;">5. Macierz pomyłek dla modelu XGBoost.</p>
        </div>
      </div>
      <p>Macierze pomyłek wskazują na wysokie wyniki poprawnego sklasyfikowania przykładów, jednak widać, że zbiór danych został dosyć zmniejszony, 
        ponieważ początkowa liczba wierszy w zbiorze danych to około 10 tys. w związku z czym zbiór testowy powienien wynosić około 2 tys., a jak można zauważyć 
        macierze pomyłek wskazują na 1189 próbek. Jest to spowodowane usunięciem wierszy z wybrakowanymi danymi (o wartościach NaN). Nie jest to jednakowoż zła 
        liczba (raczej ;P), ponieważ początkowo zastosowałam jedynie metodę usunięcia wierszy, w których były wartości NaN. Wtedy uzuskałam jedynie 585 próbek, 
        a więc zbiór danych został zmniejszony prawie 3,5 razy. Po zastosowaniu zastąpienia niektórych wartości NaN losowymi wartościami z danych kolumn zbiór został 
        znacznie mniej zmiejszony, ponieważ o około 1,7 razu.</p>
    </div>
    <div class="footer"></div>
  </div>
</body>
</html> 
'''

f = open('report.html', 'w', encoding='utf-8')
f.write(script_polish)
f.close()



























