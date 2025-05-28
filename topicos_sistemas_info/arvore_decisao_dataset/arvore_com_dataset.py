import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn import tree
import matplotlib.pyplot as plt

def create_decision_tree(
    dataset_id, # id do dataset no UCI
    dataset_name, # nome do dataset
    target_column_name='class', 
    max_total_samples=50, # limitar as amostras
    max_tree_depth=4, # Profundidade da árvore
    max_depth_tree_visualization=5 # Profundidade da árvore para visualização (imagem)
    ):

    # Verifica se o usuário deseja limitar o dataset
    if max_total_samples:
        print(f"Tentando limitar o dataset a {max_total_samples} amostras.")

    # Buscar o dataset usando o UCI
    try:
        dataset_fetched = fetch_ucirepo(id=dataset_id)
    except Exception as e:
        print(f"Erro ao buscar o dataset {dataset_name}: {e}")
        return

    features_complete = dataset_fetched.data.features
    if target_column_name in dataset_fetched.data.targets.columns:
        target_complete = dataset_fetched.data.targets[target_column_name]
    elif not dataset_fetched.data.targets.empty:
        target_complete = dataset_fetched.data.targets.iloc[:, 0]
    else:
        print(f"Erro: Não foi possível identificar a coluna alvo para {dataset_name}.")
        return

    # Limitar a quantidade de amostras (opcional)
    if max_total_samples is not None and max_total_samples < len(features_complete):
            print(f"Selecionando {max_total_samples} amostras aleatórias do dataset.")
            indices = features_complete.sample(n=max_total_samples, random_state=42).index
            features = features_complete.loc[indices]
            target_original = target_complete.loc[indices]
    else:
        features = features_complete
        target_original = target_complete
        if max_total_samples:
             print(f"O número de amostras solicitado ({max_total_samples}) é maior ou igual ao tamanho do dataset ({len(features_complete)}).")

    # Converter as classes alvo (target) de texto para números usando LabelEncoder
    target_encoder = LabelEncoder()
    target_encoded = target_encoder.fit_transform(target_original)
    original_class_names = target_encoder.classes_

    feature_names = list(features.columns)

    classifier = DecisionTreeClassifier(criterion='entropy', random_state=42, max_depth=max_tree_depth)
    classifier.fit(features, target_encoded)

    class_names_for_export = [str(cn) for cn in original_class_names]
    tree_rules = export_text(classifier, feature_names=feature_names, class_names=class_names_for_export)
    print(tree_rules)
    plt.figure(figsize=(20, 12))
    tree.plot_tree(
        classifier,
        feature_names=feature_names,
        class_names=class_names_for_export,
        filled=True,
        rounded=True,
        fontsize=10,
        max_depth=max_depth_tree_visualization
    )
    plt.title(f"Árvore de Decisão para {dataset_name} ({len(features)} amostras)", fontsize=16)
    plt.show()


# Utilizando base de dados de iris com 100 amostras aleatórias
create_decision_tree(
    dataset_id=53,
    dataset_name="Iris (Flor)",
    max_total_samples=100,
    max_tree_depth=3,
)

# Utilizando base de dados de câncer de mama com 200 amostras aleatórias
create_decision_tree(
    dataset_id=17,
    dataset_name="Câncer de Mama",
    max_total_samples=200,
    max_tree_depth=5,
)

