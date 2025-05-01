import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn import tree

# Simulando um dataset de bolas com 3 cores e características
dados_bolas = {
    'forma': ['redonda', 'redonda', 'oval', 'redonda', 'oval', 'redonda', 'redonda', 'oval', 'redonda', 'oval'],
    'tamanho': ['pequeno', 'médio', 'grande', 'grande', 'pequeno', 'médio', 'grande', 'médio', 'pequeno', 'grande'],
    'textura': ['lisa', 'lisa', 'áspera', 'lisa', 'áspera', 'áspera', 'áspera', 'lisa', 'áspera', 'lisa'],
    'cor': ['vermelha', 'vermelha', 'azul', 'verde', 'azul', 'verde', 'azul', 'vermelha', 'verde', 'vermelha']
}

# Criar DataFrame
df = pd.DataFrame(dados_bolas)

# Codificar os atributos categóricos com LabelEncoder
label_encoders = {}
for coluna in df.columns:
    le = LabelEncoder()
    df[coluna] = le.fit_transform(df[coluna])
    label_encoders[coluna] = le

# Separar atributos (X) e rótulo (y)
X = df.drop(columns=['cor'])
y = df['cor']

# Criar e treinar o classificador com entropia (ID3)
modelo = DecisionTreeClassifier(criterion='entropy', random_state=42)
modelo.fit(X, y)

# Plotar a árvore de decisão
plt.figure(figsize=(14, 8))
tree.plot_tree(
    modelo,
    feature_names=X.columns,
    class_names=label_encoders['cor'].classes_, # Use os nomes originais das cores
    filled=True,
    rounded=True
)
plt.title("Árvore de Decisão para Classificação de Bolas (baseada em Entropia)")
plt.show()