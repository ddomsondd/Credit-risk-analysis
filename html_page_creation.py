import pandas as pd
from data_preparation import data_preparation, data_preparation_one_hot
from classification import execute_classification
from regression import data_modification


data = pd.read_csv('data_atlas.csv')
data = pd.DataFrame(data)
table_view_1 = str(data.head().to_html())

data_2 = data_preparation()
data_2 = pd.DataFrame(data_2)
table_view_2 = str(data_2.head().to_html())

info = str(data_2.describe().to_html())

#GETTING DATA
one_hot_data = data_preparation_one_hot('mean')   #ZERO CHANGED TO MEAN
#one_hot_data = data_modification()               #ZERO CHANGED WITH REGRESSION
#one_hot_data = data_preparation_one_hot('knn')   #ZRO CHANGED WITH KNN

results_classification = execute_classification(one_hot_data)

results_classification = pd.DataFrame(results_classification)
results_classification_table = str(results_classification.to_html())


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
      <h2>Wstęp</h2>
      <p>Poniższy raport wykonałam, wykorzystując biblioteki takie jak:
        <ul>
          <li>Numpy</li>
          <li>Pandas</li>
          <li>Matplotlib</li>
          <li>Scikit-learn</li>
          <li>XGBoost</li>
          <li>Seaborn</li>
          <li>Imbalanced-Learn</li>
        </ul>
      </p>
      <br>

      <h2>Poznanie i obróbka danych</h2>
      <p>Pracę nad raportem zaczęłam od zapoznania się ze zbiorem danych oraz analizy danych. Wykorzystałam do tego funkcję <span style="font-style: italic;">data_check()</span>, która wyświetla nazwy nagłówków kolumn, wartości unikalne w kolumnach, a także liczbę wystąpień wartości NaN:
        <pre><code>
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
        </code></pre>
        <br>
        <br>
        Poniższa tabela przedstawia kilka pierwszych wierszy zbioru danych:
      </p>
   ''' + table_view_1 + '''
            
      <p style="font-style: italic; font-size: 14px; text-align: center; margin-top: 2px;">Tabela 1. Przedstawienie kilku pierwszych wierszy zbioru danych.</p>
      
      <br>
      <p>Tutaj mamy wybrane zbiorcze dane liczbowe:</p><table border="1" class="dataframe">
''' + info + '''<p style="font-style: italic; font-size: 14px; text-align: center; margin-top: 2px;">Tabela 2. Zbiorcze dane dla tabeli.</p>
      <br>
      <p>
          Niektóre kolumny, takie jak: 'Unnamed: 0', 'support_indicator ', nie wnoszą żadnych istotnych danych, więc można je od razu usunąć.
          Należy także przekształcić dane w niektórych kolumnach tak, aby były łatwo interpretowane przez program, np. poprzez usunięcie niepotrzebnych łańcuchów znakowych ("dzieci", "złotych"), zamianę komórek z wartościami NaN na 'unknown' albo wartości 0, czym zajmę się później.
          Zobaczyłam także, że w kolumnie 'credit_history' powinny znajdować się wartości 'brak historii', 'dobra historia' i 'zła historia', jednakże to ostatnie wogólnie nie figuruje w zbiorze, a za to są wartości NaN, więc w tym przypadku zamieniłam je na konkretną wartość.
          Wykonałam to w funkcji <span style="font-style: italic;">data_preparation()</span> w pliku <span style="font-style: italic;">data_praparation.py</span>, który wygląda następująco:
          <pre><code>
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
          </code></pre>
          <br>
          <p>Tak prezentują się dane po ich wstępnej obróbce:</p>
    ''' + table_view_2 + '''
          <p style="font-style: italic; font-size: 14px; text-align: center; margin-top: 2px;">Tabela 3. Przedstawienie kilku pierwszych wierszy zbioru danych po przetworzeniu danych.</p>
          <br>
          Następnie zamieniam komórki z kolumn 'income' oraz 'assets_value' z wartościami zero, na bardziej reprezentatywne dane. Mam do tego trzy możliwości:
          <ul>
            <li>Zamiana wartości 0 na średnią</li>
            <li>Zamiana wartości 0 na wartości wyznaczone przy użyciu algorytmu najbliższych sąsiadów (KNN)</li>
            <li>Zamiana wartości 0 na wartości wyznaczone za pomocą regresji liniowej</li>
          </ul>
          Funkcja zamiany wartości 0 na średnią:<br><br>
          <pre><code>
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
          </code></pre>
          <br><br>
          Funkcja zamiany wartości 0 na wartości wyznaczone za pomocą KNN:<br><br>
          <pre><code>            
            def change_zero_values_with_knn():
                data = data_preparation()
                imputer = KNNImputer(n_neighbors=5)
                data['income'] = data['income'].replace(0, np.nan)
                data['income'] = imputer.fit_transform(data[['income']])

                data['assets_value'] = data['assets_value'].replace(0, np.nan)
                data['assets_value'] = imputer.fit_transform(data[['assets_value']])

                return data
          </code></pre>
          <br><br>
          Funkcje zamiany wartości 0 na wartości wyznaczone za pomocą regresji liniowej:<br><br>
          <pre><code>
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
          </code></pre>
          <br><br>
          Następnym krokiem było przekształcenie kolumn na kolumny one-hot encoding. Robię to za pomocą funkcji <span style="font-style: italic;">data_preparation_one_hot()</span>, która przyjmuję jeden argument, w którym jest informacja jakiej metody chce użyć do zastąpienia wartości zerowych. Po odpowiedniej zamianie wartości zerowych wywołuję metodę <span style="font-style: italic;">get_dummies()</span> biblioteki Pandas, podając które kolumny chce przemienić na dane typu one hot encode.<br><br>
          <pre><code>
          def data_preparation_one_hot(operation):
              if operation == 'mean':
                  data = change_zero_values_to_mean()
              elif operation == 'knn':
                  data = change_zero_values_with_knn()
              else:
                  data = data_preparation()
              one_hot_data = pd.get_dummies(data, columns = ['credit_history', 'overdue_payments', 
                'employment_type', 'owns_property', 'education', 'city', 'marital_status'], dtype=int)

              return one_hot_data
          </code></pre> 
          <br><br>
          Wszelkie funkcje do obróbki danych zawarłam w pliku <span style="font-style: italic;">data_preparation.py</span>, a także <span style="font-style: italic;">regression.py</span>.
          Tak przygotowane dane można poddać dalszej analizie.
      </p>
      <br>

      <h2>Przykładowe wykresy do zbioru danych</h2>
      <p>Wszystkie wykresy utworzyłam w pliku <span style="font-style: italic;">data_visualistion.py</span>.</p>
      <div class="img_class">
        <div class="box">
          <div style="flex-direction: column;">
            <img src="plots/visualisations/age_income_pl.png" arc="" style="margin-right: 30px;">
            <p style="font-style: italic; font-size: 14px; text-align: center; margin-top: 2px;">Wykres 1.</p>
          </div>
          <div  style="width: 55%">
            Wykres przedstawia średni roczny przychód względem wieku. <br>Jak można zauważyć wykres mocno się waha, zwłaszcza dla grup wiekowych od 20 do 30 lat oraz od 50 do 70 lat. Dla przedziału wiekowego od 30 do 50 lat średni roczny dochód wynosi jednak około 23-24 tysięcy złotych.Największe wahania średniego rocznego wynagrodzenia występują po 60 roku życia - dochód zmienia się tam z 20 tysięcy do nawet 27 tysięcy złotych. 
            <br>Wykres został wykonany za pomocą następującego kodu: <br><br>
            <pre style="margin: 0; width: 100%;"><code style="font-size: medium;">
      mean_income_by_age = data.groupby('age')['income'].mean().reset_index()
      plt.figure(figsize=(8,6))
      plt.plot(mean_income_by_age['age'], mean_income_by_age['income'], color='purple', label='Średni roczny przychód', linewidth=2)
      plt.title('Średni roczny przychód w zależności od wieku')
      plt.xlabel('Wiek')
      plt.ylabel('Roczny przychód (zł)')
      plt.legend(title='Legenda')
      plt.savefig(f'plots/age_income_pl.png')
            </code></pre>
          </div>
        </div>

        <br>

        <div class="box">
          <div style="flex-direction: column;">
            <img src="plots/visualisations/children_credit_history_pl.png" arc="" style="margin-right: 30px;">
            <p style="font-style: italic; font-size: 14px; text-align: center; margin-top: 2px;">Wykres 2.</p>
          </div>
          <div style="width: 55%">
            Wykres przedstawia zestawienie liczby dzieci do historii kredytowej. Jak widać najwięcej kredytów jest brane przez osoby nie posiadające dzieci. Z każdnym kolejnym dzieckiem liczba wziętych kredytów jest mniejsza. Warto jednak zauważyć, że liczba osób z brakiem historii kredytowej bez dzieci również jest najwyższa i to w znacznym stopniu, co też świadczy o tym, iż najwięcej próbek w zbiorze stanową osoby bezdzietne.
            <br>Wykres został wykonany za pomocą następującego kodu:<br><br>
            <pre style="margin: 0; width: 100%;"><code style="font-size: medium;">
      credit_history_by_children = (
          data.groupby('children')['credit_history']
          .value_counts(normalize=True)
          .unstack()
          .reset_index()
      )

      plt.figure(figsize=(10, 6))
      sns.countplot(data=data, x='children', hue='credit_history', palette='rocket')
      plt.title('Liczba osób z różną historią kredytową w podziale na liczbę dzieci')
      plt.xlabel('Liczba dzieci')
      plt.ylabel('Liczba osób')
      plt.legend(title='Historia kredytowa', labels=['Brak historii', 'Dobra historia', 'Zła historia'])
      plt.savefig(f'plots/children_credit_history_pl.png')
            </code></pre>
          </div>
        </div>

        <br>

        <div class="box">
          <div style="flex-direction: column;">
            <img src="plots/visualisations/income_pl.png" arc="" style="margin-right: 30px;">
            <p style="font-style: italic; font-size: 14px; text-align: center; margin-top: 2px;">Wykres 3.</p>
          </div>
          <div style="width: 65%">
            Wykres przedstawia zestawienie średnich rocznych przychód względem liczby ludności otrzymujących dane wynagrodzenie. <br>W oczy od razu się rzuca wartość z zakresu około 20-25 tysięcy złotych rocznie. Liczebność osób z tym rocznym wynagrodzeniem jest największa, ponieważ wartość ta została przypisana do komórek z wartością NaN. Jest to bowiem średnia wartość wszystkich przychodów z tej kolumny. Drugą ciekawą rzeczą jest fakt, że nagle duża liczba ludzi otrzymuje roczne wynagrodzenie wynoszące 40 tysięcy złotych. Jest to nagły, duży wzrost. Na trzecim miejscu są wynagrodzenia z zakresu 15-20 tysięcy złotych rocznie.
            <br>Wykres został wykonany za pomocą następującego kodu:<br><br>
            <pre style="margin: 0; width: 100%;"><code style="font-size: medium;">
      plt.figure(figsize=(8, 6))
      sns.histplot(data['income'], bins=10, kde=True, color='#f6bcba')
      plt.title('Rozkład rocznych przychodów')
      plt.xlabel('Roczny przychód (zł)')
      plt.ylabel('Liczba ludzi')
      plt.savefig(f'plots/income_pl.png')
            </code></pre>
          </div>
        </div>
        
        <br>

        <div class="box">
          <div style="flex-direction: column;">
            <img src="plots/visualisations/employment_type_credit_risk_pl.png" arc="" style="margin-right: 30px;">
            <p style="font-style: italic; font-size: 14px; text-align: center; margin-top: 2px;">Wykres 4.</p>
          </div>
          <div style="width: 55%">
            Wykres przedstawia typy zatrudnienia w odniesieniu do ryzyka finansowego.<br>Jak widać najniższe ryzyko kredytowe jest dla ludzi, którzy mają stałą pracę, a na drugim i trzecim miejscu osoby samozatrudnione lub na określonej umowie o pracę. Co ciekawe najwyższe ryzyko kredytowe jest również dla osób ze stałą pracą, a nie jak można by przypuszczać z brakiem pracy, ale ten dysonans może być także spowodowany zdecydowanie mniejszą liczbą próbek osób bez pracy. Po tym wykresie widać również niezbalansowanie klas przedstawiających wysokie oraz niskie ryzyko kredytowe. 
            <br>Wykres został wykonany za pomocą następującego kodu:<br><br>
            <pre style="margin: 0; width: 100%;"><code style="font-size: medium;">
      plt.figure(figsize=(8, 6))
      sns.countplot(data=data, x='employment_type', hue='credit_risk', palette='rocket')
      plt.title('Typ zatrudnienia a ryzyko kredytowe')
      plt.xlabel('Typ zatrudnienia')
      plt.ylabel('Liczba osób')
      plt.legend(title='Ryzyko kredytowe', labels=['Niskie', 'Wysokie'])
      plt.savefig(f'plots/employment_type_credit_risk_pl.png')
            </code></pre>
          </div>
        </div>
      </div>
      <br>

      <h2>Macierz korelacji danych</h2>
      Stworzyłam także macierz korelacji danych liczbowych za pomocą poniższego kodu, znajdującego się w pliku <span style="font-style: italic;">data_visualisation.py</span>.
      <br><br>
      <div class="img_inline">
        <div style="flex-direction: column;">
          <img src="plots/visualisations/corr_matrix_pl.png" arc="Macierz korelacji" style="width: 500px;">
          <p style="font-style: italic; font-size: 14px; text-align: center; margin-top: 2px;">Wykres 5. Macierz korelacji.</p>
        </div>
      </div>
      <p>Można zauważyć, że dane są ze sobą słabo skorelowane, wyróżnia się kolumna 'active_loans', której korelacja wynosi około 0.31 do kolumny 'credit_risk'. Oznacza to, że może ona mieć największy wpływ na wartość predykcji tej zmiennej spośród cech znajdujących się w tym zestawieniu. Macierz ta pokazuje jednak brak wyraźnych zależności między tymi zmiennymi. Pokazuje to pole do możliwej dalszej transformacji danych, ale to zagadnienie na późniejszy czas.</p>
      <br>

      <h2>Modele klasyfikacyjne</h2>
      <p>Tworząc modele klasyfikacyjne założyłam, że przewidywaną wartością będzie ryzyko kredytowe (kolumna 'credit_risk').<br>Porównałam ze sobą pięć modeli klasyfikacyjnych. Modele stworzyłam w pliku <span style="font-style: italic;">classification.py</span>.
        <ul>
          <li>Decision Tree Classifier</li>
          <li>Random Forest Classifier</li>
          <li>Support Vector Classifier</li>
          <li>Gaussian Naive Bayes</li> 
          <li>XGBoost Classifier</li>
        </ul>

        W funkcji <span style="font-style: italic;">execute_classification()</span> dzielę zbiór na dane X i y, czyli dane wejściowe niezależne, z których model ma się uczyć predykcji oraz na dane zależne, które mają być przewidywane - w tym przypadku zbiór y stanowi kolumna <span style="font-style: italic;">'credit_risk'</span>, ponieważ chce przewidywać wysokie bądź niskie ryzyko kredytowe. Następnie sprawdzam rozkład próbek w tej klasie i dostaję wynik 9675 dla klasy 0 stanowiącej niskie ryzyko kredytowe do 325 dla klasy 1, stanowiącej wysokie ryzyko kredytowe. Jak widać klasy są mocno niezbalansowane. Prezentuje to poniższy wykres:
        <div class="img_inline">
          <div style="flex-direction: column;">
            <img src="plots/more_infos/sample_distribution_pl.png" arc="Rozkład klas" style="width: 500px;">
            <p style="font-style: italic; font-size: 14px; text-align: center; margin-top: 2px;">Wykres 6. Rozkład klas wartości przewidywanych.</p>
          </div>
        </div>
        <br>
        Aby temu zaradzić - balansuję klasy za pomocą metody SMOTE, która na podstawie istniejącego zbioru danych dodaje próbki do klasy, która jest mniej liczna. Po wykonaniu tej operacji próbki wynoszą tyle samo, czyli 9675 próbek, co jest widoczne na wykresie:
        <div class="img_inline">
          <div style="flex-direction: column;">
            <img src="plots/more_infos/sample_distribution_smote_pl.png" arc="Rozkład klas po SMOTE" style="width: 500px;">
            <p style="font-style: italic; font-size: 14px; text-align: center; margin-top: 2px;">Wykres 7. Rozkład klas wartości przewidywanych po zastosowaniu metody SMOTE.</p>
          </div>
        </div>
        <br>Następnie definiuje jakie modele chce stworzyć i w pętli wywołuję metodę <span style="font-style: italic;">create_classification_model()</span>, która tworzy dane modele i zwraca metryki dokładności, czułości, specyficzności, F1 oraz Cohen Kappa. Wywołuję także metodę tworzącą macierze błedów, ale więcej o tym w dalszej części raportu.<br>
        <br>
        <pre><code>
          def execute_classification(one_hot_data):
              X = one_hot_data.drop('credit_risk', axis=1)
              y = one_hot_data['credit_risk']

              #checking samples distribution
              count_class = y.value_counts()
              #print(count_class)
              #english version
              plt.bar(count_class.index, count_class.values)
              plt.xlabel('Class')
              plt.ylabel('Count')
              plt.title('Class Distribution')
              plt.xticks(count_class.index, ['Class 0', 'Class 1'])
              plt.savefig(f'plots/sample_distribution_en.png', bbox_inches='tight')

              #polish version
              plt.bar(count_class.index, count_class.values)
              plt.xlabel('Klasa')
              plt.ylabel('Liczba próbek')
              plt.title('Rozkład klas')
              plt.xticks(count_class.index, ['Klasa 0', 'Klasa 1'])
              plt.savefig(f'plots/sample_distribution_pl.png', bbox_inches='tight')


              #class balancing
              smote=SMOTE(sampling_strategy='minority') 
              X,y=smote.fit_resample(X,y)
              #print(y.value_counts())


              models = [DecisionTreeClassifier(), RandomForestClassifier(), SVC(), GaussianNB(), XGBClassifier()]
              models_names = ['Decision Tree Classifier', 'Random Forest Classifier', 'Support Vector Classifier', 
                                'Gausian Naive Bayes', 'XGBoost Classifier']
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
        </code></pre>
        <br><br>
        Funkcja tworząca modele oraz metryki wygląda następująco:<br><br>
        <pre><code>
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
                  plt.ylabel("Features")
                  plt.savefig(f'plots/features_importance_rfc_en.png', bbox_inches='tight')

                  plt.figure(figsize=(10, 6))
                  plt.title("Ważność cech")
                  plt.bar(range(X.shape[1]), importances[indices], color='purple')
                  plt.xticks(range(X.shape[1]), X, rotation=90)
                  plt.xlabel("Ważność")
                  plt.ylabel("Cecha")
                  plt.savefig(f'plots/features_importance_rfc_pl.png', bbox_inches='tight')

              
              plot_confmat(y_test, y_pred, model_name)

              return accuracy, recall, specificity, f1, kappa
        </code></pre>
        <br><br>


        <h3>Ważność cech</h3>
        Warto zauważyć, że jeśli modelem jest las losowy, to tworzę wykres ważności cech, który definiuje stopień wpływu danej cechy na wynik predykcji. Zestawienie wygląda następująco:
        <br><br>
        <div class="img_inline">
          <div style="flex-direction: column;">
            <img src="plots/more_infos/features_importance_rfc_pl.png" arc="Ważność cech" style="width: 750px;">
            <p style="font-style: italic; font-size: 14px; text-align: center; margin-top: 2px;">Wykres 8. Ważność cech.</p>
          </div>
        </div>
        <br>
        Muszę jednak nad tym więcej popracować, ponieważ nie wiem czy jest to faktycznie miarodajne, gdyż cechy te są po prostu ustawione w kolejności takiej jak występują w tabeli z danymi, a więc ciężko stwierdzić czy faktycznie ich ważność przebiega w ten sposób.
        <br><br></p>

        <h3>Metryki dokładności dla poszczególnych modeli</h3>
        <p>Przechodząc do metryk modeli - jak widać po poniższej tabeli najlepiej dopasowane modele to las losowy oraz XGBoost. Nieco gorzej, ale w dalszym ciągu bardzo dobrze wypada model drzewa decyzyjnego. Te trzy modele osiągają dobre metryki zarówno dokładności, czułości i specyficzności, jak i bardziej dokładnych metryk, jakimi są F1 Score, a także Cohen Kappa. Można zatem stwierdzić, że dobrze przewidują wartość badaną. Gorzej przedstawiają się metryki maszyny wektorów nośnych oraz naiwnego klasyfikatora Bayesa. W tych przypadkach modele źle predyują dane, nad czym można spróbować popracować w późniejszym czasie. </p>
    ''' + results_classification_table + '''
            <p style="font-style: italic; font-size: 14px; text-align: center; margin-top: 2px;">Tabela 4. Przedstawienie miar jakości dopasowania danych modeli.</p>
            <br>

      <h3>Macierze pomyłek dla poszczególnych modeli</h3>
      <p>Macierze pomyłek stworzyłam za pomocą metody <span style="font-style: italic;">plot_confmat()</span>:</p>
      <pre><code>
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
      </code></pre>
      <br>
      <p>Prezentują się one następująco:</p>
      <div class="img_inline">
        <div style="flex-direction: column;">
          <img src="plots/models/plot_class_Decision Tree Classifier_pl.png" arc="Confusion matrix for Decision Tree Classifier">
          <p style="font-style: italic; font-size: 14px; text-align: center; margin-top: 2px;">1. Macierz pomyłek dla modelu drzewa decyzyjnego.</p>
        </div>
        <div style="flex-direction: column;">
          <img src="plots/models/plot_class_Random Forest Classifier_pl.png" arc="Confusion matrix for Random Forest Classifier">
          <p style="font-style: italic; font-size: 14px; text-align: center; margin-top: 2px;">2. Macierz pomyłek dla modelu lasu losowego.</p>
        </div>
        <div style="flex-direction: column;">
          <img src="plots/models/plot_class_Support Vector Classifier_pl.png" arc="Confusion matrix for Support Vector Machine Classifier">
          <p style="font-style: italic; font-size: 14px; text-align: center; margin-top: 2px;">3. Macierz pomyłek dla modelu maszyny wektorów nośnych.</p>
        </div>
        <div style="flex-direction: column;">
          <img src="plots/models/plot_class_Gausian Naive Bayes_pl.png" arc="Confusion matrix for Gaussian Naive Bayes Classifier">
          <p style="font-style: italic; font-size: 14px; text-align: center; margin-top: 2px;">4. Macierz pomyłek dla naiwnego klasyfikatora Bayesa.</p>
        </div>
        <div style="flex-direction: column;">
          <img src="plots/models/plot_class_XGBoost Classifier_pl.png" arc="Confusion matrix for XGBoost Classifier">
          <p style="font-style: italic; font-size: 14px; text-align: center; margin-top: 2px;">5. Macierz pomyłek dla modelu XGBoost.</p>
        </div>
      </div>
      <p>Macierze pomyłek dla modeli lasu losowego, XGBoost oraz drzewa decyzyjnego wskazują na wysokie wyniki poprawnego sklasyfikowania przykładów do zbiorów true-positive oraz true-negative, co jest dobrym wynikiem. Występują drobne odchylenia zwłaszcza w drzewie decyzyjnym, w którym 18 próbek zostało sklastfikowanych do false-positive, a 9 do false-negative. W przypadku lasu losowego było to 16 błędnie przewidzianych próbek, a w modelu XGBoost 10.
        <br><br>Zdecydowanie gorzej wygląda sytuacja dla maszyny wektorów nośnych oraz naiwnego klasyfikatora Bayesa. W tych przypadkach zostały osiągnięte bardzo słabe wyniki dotyczące predykcji.</p>

      <br>
      <h2>PODSUMOWANIE</h2>
      <p>Udało mi się stworzyć modele przewidujące wystąpienie lub brak wystąpienia ryzyka kredytowego. Najlepszymi modelami jest las losowy, XGBoost oraz drzewo decyzyjne. Maszyna wektorów nośnych i naiwny klasyfikator Bayesa osiągają słabe wyniki predykcji. Jest to pole do rozwoju projektu w przyszłości. Działanie poszczególnych modeli jednakowoż znacznie się poprawiło, ponieważ w poprzedniej wersji raportu macierze pomyłek wskazywały na prawie 99% dopasowania do zbioru true-positive, jednakże modele te były niereprezentatywne, ze względu na brak zbalansowania klas. Teraz macierze pomyłek wskazują w najlepszych modelach dopasowanie predykcji do zbiorów true-positive oraz true-negative i stosunku praktycznie 50% do 50%, co jest oczekiwanym wynikiem. 
        <br><br>Tworząc projekt zaimplementowałam trzy wersje wypełnienia wartości NaN, występujących w kolumnach z danymi liczbowymi. Jest to wypełnienie średnią, regresją oraz algorytmem najbliższych sąsiadów. W finalniej wersji raportu korzystam jednakże z opcji pierwszej, ponieważ jest ona optymalna, a dodatkowo nie występują zauważnalne różnice między działaniem modeli przy wykorzystaniu każdej z opcji. Przetestowałam wszystkie i wyniki wychodzą bardzo zbliżone. Choć w dalszym ciągu nie jestem do końca przekonana czy są to dobre rozwiązania, ponieważ próbki zostają mocno podbite przez jedną wartość, jaką jest średnia. Można się zastanowić czy jest lepsza metoda na zastąpienie wartości NaN w tych kolumnach.
        <br><br>Polem do przyszłego rozwoju może być także transformacja danych, aby były one bardziej skorelowane, a także poprawie ważności cech. 
        <br><br>Raport jednak został poprawiony pod względem ilości próbek, ponieważ we wcześniejszej wersji usunęłam znaczną część zbiori, poprzez użycie funkcji dropna(). W tym odpowiednio zastąpiłam brakujące wartości, a także zbalansowałam klasy zbioru predykcyjnego y, tak aby wynosiły od 50% do 50%, toteż uzyskałam nawet większy zbiór danych.  
        <br><br>Polem do rozwoju w przyszłości jest także wizualizacja danych, ponieważ można się zastanowić nad większą ilością wykresów obrazujących dane, pasujących do tematu raportu.
        <br><br>Raport można także rozwijać o wykresy probabilistyczne, a także analizę kwantyli.
      </p>
    </div>
    <div class="footer"></div>
  </div>
</body>
</html> 
'''

f = open('raport_pl.html', 'w', encoding='utf-8')
f.write(script_polish)
f.close()


#English version
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
      <h1>Creditworthiness Analysis</h1>
    </div>
    
    <div class="main">
      <h2>Introduction</h2>
      <p>This report was created using libraries such as:
        <ul>
          <li>Numpy</li>
          <li>Pandas</li>
          <li>Matplotlib</li>
          <li>Scikit-learn</li>
          <li>XGBoost</li>
          <li>Seaborn</li>
          <li>Imbalanced-Learn</li>
        </ul>
      </p>
      <br>
    
      <h2>Exploration and Data Preparation</h2>
      <p>I started working on the report by exploring the dataset and performing data analysis. I used the <span style="font-style: italic;">data_check()</span> function, which displays column header names, unique values in the columns, and the count of NaN values:</p>
    <pre><code>
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
        </code></pre>
        <br>
        <br>
        The table below shows the first few rows of the dataset:
      </p>
   ''' + table_view_1 + '''            
<p style="font-style: italic; font-size: 14px; text-align: center; margin-top: 2px;">Table 1. Display of the first few rows of the dataset.</p>

<br>
<p>Here are the selected summary numerical data:</p>
''' + info + '''
<p style="font-style: italic; font-size: 14px; text-align: center; margin-top: 2px;">Table 2. Summary data for the table.</p>
<br>
<p>
    Some columns, such as 'Unnamed: 0', 'support_indicator', do not provide any significant data, so they can be removed right away. 
    Data in certain columns also needs to be transformed to be easily interpreted by the program, for example, by removing unnecessary strings like ("children", "zlotych"), replacing NaN cells with 'unknown' or zero values, which I will address later. 
    I also noticed that the 'credit_history' column should contain values like 'no history', 'good history', and 'bad history', however, the last one is missing from the dataset, and instead, there are NaN values, which I replaced with a specific value.
    I performed this in the <span style="font-style: italic;">data_preparation()</span> function in the <span style="font-style: italic;">data_preparation.py</span> file, which looks as follows:
</p>
 <pre><code>
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
          </code></pre>
          <br>
          <p>This is how the data looks after its initial processing:</p>
    ''' + table_view_2 + '''
<p style="font-style: italic; font-size: 14px; text-align: center; margin-top: 2px;">Table 3. Display of the first few rows of the dataset after data processing.</p>
<br>
Next, I replace the cells in the 'income' and 'assets_value' columns with zero values with more representative data. I have three options for this:
<ul>
  <li>Replacing zero values with the mean</li>
  <li>Replacing zero values with values determined using the KNN algorithm</li>
  <li>Replacing zero values with values determined by linear regression</li>
</ul>
Function for replacing zero values with the mean:<br><br>

          <pre><code>
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
          </code></pre>
          <br><br>
          Function for replacing zero values with values determined by KNN:<br><br>
          <pre><code>            
            def change_zero_values_with_knn():
                data = data_preparation()
                imputer = KNNImputer(n_neighbors=5)
                data['income'] = data['income'].replace(0, np.nan)
                data['income'] = imputer.fit_transform(data[['income']])

                data['assets_value'] = data['assets_value'].replace(0, np.nan)
                data['assets_value'] = imputer.fit_transform(data[['assets_value']])

                return data
          </code></pre>
          <br><br>
          Functions for replacing zero values with values determined by linear regression:<br><br>
          <pre><code>
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
          </code></pre>
          <br><br>
          The next step was to transform the columns into one-hot encoded columns. I do this using the <span style="font-style: italic;">data_preparation_one_hot()</span> function, which takes one argument specifying the method to use for replacing zero values. After appropriately replacing the zero values, I call the <span style="font-style: italic;">get_dummies()</span> method from the Pandas library, specifying which columns I want to transform into one-hot encoded data.<br><br>
          <pre><code>
          def data_preparation_one_hot(operation):
              if operation == 'mean':
                  data = change_zero_values_to_mean()
              elif operation == 'knn':
                  data = change_zero_values_with_knn()
              else:
                  data = data_preparation()
              one_hot_data = pd.get_dummies(data, columns = ['credit_history', 'overdue_payments', 
                'employment_type', 'owns_property', 'education', 'city', 'marital_status'], dtype=int)

              return one_hot_data
          </code></pre> 
          <br>
          <p>All data processing functions are contained in the <span style="font-style: italic;">data_preparation.py</span> and <span style="font-style: italic;">regression.py</span> files. The prepared data can then be subjected to further analysis.</p>
          <br>
          
          <h2>Sample Charts for the Dataset</h2>
          <p>All the charts were created in the <span style="font-style: italic;">data_visualistion.py</span> file.</p>
          <div class="img_class">
            <div class="box">
              <div style="flex-direction: column;">
                <img src="plots/visualisations/age_income_en.png" arc="" style="margin-right: 30px;">
                <p style="font-style: italic; font-size: 14px; text-align: center; margin-top: 2px;">Figure 1.</p>
              </div>
              <div style="width: 55%">
                The chart shows the average annual income in relation to age. <br>As can be seen, the chart fluctuates significantly, especially for age groups from 20 to 30 years and from 50 to 70 years. For the age range of 30 to 50 years, the average annual income is around 23-24 thousand PLN. The largest fluctuations in average annual income occur after the age of 60, where income changes from 20 thousand to even 27 thousand PLN.<br>
                The chart was created using the following code: <br><br>
          
            <pre style="margin: 0; width: 100%;"><code style="font-size: medium;">
      plt.figure(figsize=(8,6))
      plt.plot(mean_income_by_age['age'], mean_income_by_age['income'], color='purple', label='Średni roczny przychód', linewidth=2)
      plt.title('Average annual income depending on age')
      plt.xlabel('Age')
      plt.ylabel('Annual income (zł)')
      plt.legend(title='Legend', labels=['Average annual income'])
      #plt.show()
      plt.savefig(f'plots/visualisations/age_income_en.png')
            </code></pre>
          </div>
        </div>

        <br>

        <div class="box">
          <div style="flex-direction: column;">
            <img src="plots/visualisations/children_credit_history_en.png" arc="" style="margin-right: 30px;">
            <p style="font-style: italic; font-size: 14px; text-align: center; margin-top: 2px;">Figure 2.</p>
          </div>
          <div style="width: 55%">
            The chart shows a comparison of the number of children with credit history. As seen, the most loans are taken by people without children. With each additional child, the number of loans decreases. However, it is worth noting that the number of people without a credit history who do not have children is also the highest, and by a significant margin, which indicates that the majority of samples in the dataset are childless individuals.<br>
            The chart was created using the following code:<br><br>          
            <pre style="margin: 0; width: 100%;"><code style="font-size: medium;">
      credit_history_by_children = (
          data.groupby('children')['credit_history']
          .value_counts(normalize=True)
          .unstack()
          .reset_index()
      )

      plt.figure(figsize=(10, 6))
      sns.countplot(data=data, x='children', hue='credit_history', palette='rocket')
      plt.title('Number of people with different credit histories divided by the number of children')
      plt.xlabel('Number of children')
      plt.ylabel('Number of people')
      plt.legend(title='Credit history', labels=['No history', 'Good history', 'Bad history'])
      plt.savefig(f'plots/visualisations/children_credit_history_en.png')
            </code></pre>
          </div>
        </div>

        <br>

        <div class="box">
          <div style="flex-direction: column;">
            <img src="plots/visualisations/income_en.png" arc="" style="margin-right: 30px;">
            <p style="font-style: italic; font-size: 14px; text-align: center; margin-top: 2px;">Figure 3.</p>
            </div>
            <div style="width: 65%">
              The chart shows a comparison of average annual income relative to the number of people receiving each specific salary.<br>Immediately noticeable is the value in the range of approximately 20-25 thousand PLN per year. The number of people with this annual income is the highest because this value was assigned to cells with NaN values. This is the average value of all incomes in that column. Another interesting point is that a suddenly large number of people earn an annual salary of 40 thousand PLN, showing a sharp increase. In third place are salaries in the range of 15-20 thousand PLN per year.<br>
              The chart was created using the following code:<br><br>            
            <pre style="margin: 0; width: 100%;"><code style="font-size: medium;">
      plt.figure(figsize=(8, 6))
      sns.histplot(data['income'], bins=10, kde=True, color='#f6bcba')
      plt.title('Distribution of annual income')
      plt.xlabel('Annual income (zł)')
      plt.ylabel('Number of people')
      plt.savefig(f'plots/visualisations/income_en.png')
            </code></pre>
          </div>
        </div>
        
        <br>

        <div class="box">
          <div style="flex-direction: column;">
            <img src="plots/visualisations/employment_type_credit_risk_en.png" arc="" style="margin-right: 30px;">
            <p style="font-style: italic; font-size: 14px; text-align: center; margin-top: 2px;">Figure 4.</p>
          </div>
          <div style="width: 55%">
            The chart shows employment types in relation to financial risk.<br>As seen, the lowest credit risk is for people with permanent jobs, and in second and third place are self-employed individuals or those on fixed-term contracts. Interestingly, the highest credit risk is also for people with permanent jobs, rather than those without work, which could be due to the significantly smaller sample size of unemployed individuals. This chart also highlights the imbalance between classes representing high and low credit risk.<br>
            The chart was created using the following code:<br><br>          
            <pre style="margin: 0; width: 100%;"><code style="font-size: medium;">
      plt.figure(figsize=(8, 6))
      sns.countplot(data=data, x='employment_type', hue='credit_risk', palette='rocket')
      plt.title('Employment type to credit risk')
      plt.xlabel('Employment type')
      plt.ylabel('Number of people')
      plt.legend(title='Credit risk', labels=['Low', 'High'])
      plt.savefig(f'plots/visualisations/employment_type_credit_risk_en.png')
            </code></pre>
          </div>
        </div>
      </div>
      <br>

      <h2>Data Correlation Matrix</h2>
      <p>I also created a correlation matrix for the numerical data using the code below, located in the <span style="font-style: italic;">data_visualisation.py</span> file.</p>
      <div class="img_inline">
        <div style="flex-direction: column;">
          <img src="plots/visualisations/corr_matrix_en.png" alt="Correlation Matrix" style="width: 500px;">
          <p style="font-style: italic; font-size: 14px; text-align: center; margin-top: 2px;">Figure 5. Correlation Matrix.</p>
        </div>
      </div>
      <p>You can observe that the data is weakly correlated with each other, with the 'active_loans' column standing out, which has a correlation of about 0.31 with the 'credit_risk' column. This indicates that it may have the greatest influence on the prediction of this variable compared to the other features in this dataset. However, this matrix shows a lack of clear dependencies between the variables. This suggests an opportunity for further data transformation, but that is a topic for later.</p>
      <br>
      
      <h2>Classification Models</h2>
      <p>When creating the classification models, I assumed that the target value would be credit risk (the 'credit_risk' column).<br>I compared five classification models. The models were created in the <span style="font-style: italic;">classification.py</span> file.
        <ul>
          <li>Decision Tree Classifier</li>
          <li>Random Forest Classifier</li>
          <li>Support Vector Classifier</li>
          <li>Gaussian Naive Bayes</li> 
          <li>XGBoost Classifier</li>
        </ul>
      
      In the <span style="font-style: italic;">execute_classification()</span> function, I split the dataset into independent input data X, from which the model will learn predictions, and dependent data y, which should be predicted – in this case, the y set consists of the <span style="font-style: italic;">'credit_risk'</span> column, as I want to predict high or low credit risk. I then check the distribution of samples in this class, and I get 9675 samples for class 0 (low credit risk) and 325 for class 1 (high credit risk). As can be seen, the classes are highly imbalanced. This is presented in the chart below:
      <br><br>
      <div class="img_inline">
        <div style="flex-direction: column;">
          <img src="plots/more_infos/sample_distribution_en.png" alt="Class Distribution" style="width: 500px;">
          <p style="font-style: italic; font-size: 14px; text-align: center; margin-top: 2px;">Figure 6. Distribution of predicted class values.</p>
        </div>
      </div>
      <br>
      To address this issue, I balance the classes using the SMOTE method, which adds samples to the underrepresented class based on the existing dataset. After performing this operation, both classes have an equal number of samples, which is 9675, as shown in the following chart:
      <br><br>
      <div class="img_inline">
        <div style="flex-direction: column;">
          <img src="plots/more_infos/sample_distribution_smote_en.png" alt="Class Distribution after SMOTE" style="width: 500px;">
          <p style="font-style: italic; font-size: 14px; text-align: center; margin-top: 2px;">Figure 7. Distribution of predicted class values after applying the SMOTE method.</p>
        </div>
      </div>
      <br>Next, I define which models I want to create, and in a loop, I call the <span style="font-style: italic;">create_classification_model()</span> method, which creates these models and returns accuracy, sensitivity, specificity, F1, and Cohen Kappa metrics. I also call the method that generates confusion matrices, but more on that later in the report.<br>
      <br>
      
        <pre><code>
          def execute_classification(one_hot_data):
              X = one_hot_data.drop('credit_risk', axis=1)
              y = one_hot_data['credit_risk']

              #checking samples distribution
              count_class = y.value_counts()
              #print(count_class)
              #english version
              plt.bar(count_class.index, count_class.values)
              plt.xlabel('Class')
              plt.ylabel('Count')
              plt.title('Class Distribution')
              plt.xticks(count_class.index, ['Class 0', 'Class 1'])
              plt.savefig(f'plots/sample_distribution_en.png', bbox_inches='tight')

              #polish version
              plt.bar(count_class.index, count_class.values)
              plt.xlabel('Klasa')
              plt.ylabel('Liczba próbek')
              plt.title('Rozkład klas')
              plt.xticks(count_class.index, ['Klasa 0', 'Klasa 1'])
              plt.savefig(f'plots/sample_distribution_pl.png', bbox_inches='tight')


              #class balancing
              smote=SMOTE(sampling_strategy='minority') 
              X,y=smote.fit_resample(X,y)
              #print(y.value_counts())


              models = [DecisionTreeClassifier(), RandomForestClassifier(), SVC(), GaussianNB(), XGBClassifier()]
              models_names = ['Decision Tree Classifier', 'Random Forest Classifier', 'Support Vector Classifier', 
                                'Gausian Naive Bayes', 'XGBoost Classifier']
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
        </code></pre>
        <br><br>
        The function that creates models and metrics:<br><br>
        <pre><code>
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
                  plt.ylabel("Features")
                  plt.savefig(f'plots/features_importance_rfc_en.png', bbox_inches='tight')

                  plt.figure(figsize=(10, 6))
                  plt.title("Ważność cech")
                  plt.bar(range(X.shape[1]), importances[indices], color='purple')
                  plt.xticks(range(X.shape[1]), X, rotation=90)
                  plt.xlabel("Ważność")
                  plt.ylabel("Cecha")
                  plt.savefig(f'plots/features_importance_rfc_pl.png', bbox_inches='tight')

              
              plot_confmat(y_test, y_pred, model_name)

              return accuracy, recall, specificity, f1, kappa
        </code></pre>
        <br><br>


        <h3>Feature Importance</h3>
        <p>It is worth noting that when the model is a Random Forest, I create a feature importance chart that defines the degree of influence each feature has on the prediction outcome. The chart is as follows:</p>
        
        <div class="img_inline">
          <div style="flex-direction: column;">
            <img src="plots/more_infos/features_importance_rfc_en.png" alt="Feature Importance" style="width: 750px;">
            <p style="font-style: italic; font-size: 14px; text-align: center; margin-top: 2px;">Figure 8. Feature Importance.</p>
          </div>
        </div>
    
        <p>However, I need to work on this further, as I am unsure if it is truly reliable, since these features are simply listed in the order they appear in the data table. Therefore, it is difficult to determine whether their importance actually follows this sequence.</p>
        <br>
        
        <h3>Accuracy Metrics for Individual Models</h3>
        <p>Moving on to the model metrics - as shown in the table below, the best-fitting models are the Random Forest and XGBoost. The Decision Tree model performs slightly worse, but still very well. These three models achieve good metrics for accuracy, sensitivity, specificity, as well as more precise metrics such as F1 Score and Cohen Kappa. Therefore, it can be concluded that they predict the target variable well. On the other hand, the metrics for the Support Vector Machine and Naive Bayes Classifier are worse. In these cases, the models perform poorly in terms of prediction, and improvements could be explored in the future.</p>
        ''' + results_classification_table + '''
      <p style="font-style: italic; font-size: 14px; text-align: center; margin-top: 2px;">Table 4. Presentation of the quality measures of the fitting of these models.</p>
      <br>

      <h3>Confusion Matrices for Individual Models</h3>
      <p>The confusion matrices were created using the <span style="font-style: italic;">plot_confmat()</span> method:</p>
      <pre><code>
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
      </code></pre>
      <br>
      <p>The results are as follows:</p>
      <div class="img_inline">
        <div style="flex-direction: column;">
          <img src="plots/models/plot_class_Decision Tree Classifier_en.png" arc="Confusion matrix for Decision Tree Classifier">
          <p style="font-style: italic; font-size: 14px; text-align: center; margin-top: 2px;">1. Confusion matrix for Decision Tree Model.</p>
        </div>
        <div style="flex-direction: column;">
          <img src="plots/models/plot_class_Random Forest Classifier_en.png" arc="Confusion matrix for Random Forest Classifier">
          <p style="font-style: italic; font-size: 14px; text-align: center; margin-top: 2px;">2. Confusion matrix for Random Forest Model.</p>
        </div>
        <div style="flex-direction: column;">
          <img src="plots/models/plot_class_Support Vector Classifier_en.png" arc="Confusion matrix for Support Vector Machine Classifier">
          <p style="font-style: italic; font-size: 14px; text-align: center; margin-top: 2px;">3. Confusion matrix for Support Vector Machine Model.</p>
        </div>
        <div style="flex-direction: column;">
          <img src="plots/models/plot_class_Gausian Naive Bayes_en.png" arc="Confusion matrix for Gaussian Naive Bayes Classifier">
          <p style="font-style: italic; font-size: 14px; text-align: center; margin-top: 2px;">4. Confusion matrix for Gaussian Naive Bayes Model.</p>
        </div>
        <div style="flex-direction: column;">
          <img src="plots/models/plot_class_XGBoost Classifier_en.png" arc="Confusion matrix for XGBoost Classifier">
          <p style="font-style: italic; font-size: 14px; text-align: center; margin-top: 2px;">5. Confusion matrix for XGBoost Model.</p>
        </div>
      </div>
      <p>The confusion matrices for the Random Forest, XGBoost, and Decision Tree models indicate high accuracy in classifying examples into the true-positive and true-negative sets, which is a good result. There are minor deviations, especially in the Decision Tree, where 18 samples were classified as false-positive and 9 as false-negative. In the case of Random Forest, there were 16 misclassified samples, and in the XGBoost model, 10.
        <br><br>The situation is significantly worse for the Support Vector Machine and Naive Bayes Classifier. In these cases, very poor prediction results were achieved.</p>

      <br>
      <h2>SUMMARY</h2>
      <p>I successfully created models predicting the occurrence or absence of credit risk. The best-performing models are Random Forest, XGBoost, and Decision Tree. Support Vector Machine and Naive Bayes Classifier, on the other hand, achieve poor prediction results, which creates room for improvement in future iterations of the project. However, the performance of all models has significantly improved, as confusion matrices in the previous version of the report indicated almost 99% alignment with the true-positive set. These models, however, were not representative due to the lack of balanced classes. Now, the confusion matrices of the best models show alignment of predictions with the true-positive and true-negative sets and a nearly 50%-50% balance, which is the desired outcome.
          <br><br>While working on this project, I implemented three methods for filling NaN values in numerical columns: filling with the mean, regression, and the k-nearest neighbors algorithm. In the final version of the report, I used the first option because it is the most optimal and there are no noticeable differences in the performance of the models using any of the methods. I tested all three approaches, and the results are very similar. Nevertheless, I am still not entirely convinced whether these are the best solutions, as the samples are heavily influenced by a single value, the mean. It might be worth considering whether there is a better method for replacing NaN values in these columns.
          <br><br>A future area of development could include data transformation to increase correlation and improve feature importance.
          <br><br>The report has also been improved regarding the number of samples. In the earlier version, I removed a significant part of the dataset using the dropna() function. This time, I replaced the missing values appropriately and balanced the prediction dataset `y` so that the classes are split 50%-50%. As a result, I obtained an even larger dataset.
          <br><br>Another area for future development is data visualization, as it may be worth considering more charts representing the data, tailored to the report's topic.
          <br><br>The report could also be extended with probabilistic plots and quantile analysis.
      </p>

      </p>
    </div>
    <div class="footer"></div>
  </div>
</body>
</html> 
'''

f = open('report_en.html', 'w', encoding='utf-8')
f.write(script_english)
f.close()
