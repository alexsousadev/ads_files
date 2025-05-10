import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn import tree

# Utilizando os atributos Orçamento, Preferência de comida e Distância.
preferences_and_Restritions = {
    'orcamento': ['baixo', 'médio', 'médio', 'alto','baixo', 'médio', 'alto', 'baixo'],
    'preferencia_comida': ['Fast Food', 'Italiano', 'Japonês', 'Italiano', 'Japonês', 'Fast Food', 'Fast Food', 'Italiano'],
    'distancia': ['Próximo', 'Próximo', 'Longe', 'Longe', 'Próximo', 'Próximo', 'Longe', 'Longe'],
    'recomendar': [True, True, False, False, True, True, False, True]
}

# Criar DataFrame com os dados de preferências e a decisão de recomendação
df = pd.DataFrame(preferences_and_Restritions)

# Codificar os atributos categóricos com LabelEncoder
label_encoders = {}
for coluna in df.columns:
    le = LabelEncoder()
    df[coluna] = le.fit_transform(df[coluna])
    label_encoders[coluna] = le

# Separar atributos (X) e rótulo (y)
X = df[['orcamento', 'preferencia_comida', 'distancia']]
y = df['recomendar']

# Criar e treinar o classificador com entropia (ID3)
modelo = DecisionTreeClassifier(criterion='entropy', random_state=42)
modelo.fit(X, y)

# Plotar a árvore de decisão
plt.figure(figsize=(14, 8))
tree.plot_tree(
    modelo,
    feature_names=X.columns,
    class_names=['Não Recomendar', 'Recomendar'], # Nomes das classes para a recomendação
    filled=True,
    rounded=True
)
plt.title("Árvore de Decisão para Recomendação de Restaurantes")
plt.show()