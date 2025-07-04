import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# Definir nomes das colunas
columns = ['area', 'perimeter', 'compactness', 'length_kernel', 'width_kernel', 
           'asymmetry_coefficient', 'length_kernel_groove', 'target']

# Carregar dados do arquivo seeds_dataset.txt
seeds = pd.read_csv('seeds_dataset.txt', sep='\s+', header=None, names=columns)

# Separar features e target
X = seeds.drop('target', axis=1)
y = seeds['target']

# Dividir entre treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Criar e treinar o modelo Naive Bayes
modelo = GaussianNB()
modelo.fit(X_train, y_train)

# Fazer previsões
y_pred = modelo.predict(X_test)

# Avaliar o modelo
print("Acurácia:", accuracy_score(y_test, y_pred))
print("\nRelatório de Classificação:\n", classification_report(y_test, y_pred, 
target_names=['Kama', 'Rosa', 'Canadian']))