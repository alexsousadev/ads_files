from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
# Carregar dataset Wine
wine = load_wine()
X = wine.data
y = wine.target
# Dividir entre treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Criar o modelo Naive Bayes
modelo = GaussianNB()
# Treinar o modelo
modelo.fit(X_train, y_train)
# Fazer previsões
y_pred = modelo.predict(X_test)
# Avaliar o modelo
print("Acurácia:", accuracy_score(y_test, y_pred))
print("\nRelatório de Classificação:\n", classification_report(y_test, y_pred, 
target_names=wine.target_names))