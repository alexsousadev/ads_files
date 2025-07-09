
# Exercícios propostos

1. Teste diferentes valores de K (de 1 a 15) e observe como a acurácia muda.
    > Apliquei alguns valores diferentes de K para testar como a acurácia iria alterar, testando os valores [1](./1_testes_knn/knn_1.py), [2](./1_testes_knn/knn_2.py) e [12](./1_testes_knn/knn_12.py). Em geral, valores muito baixos de K podem causar uma acurácia menor devido à sensibilidade a variações nos dados. À medida que K aumenta, a acurácia tende a estabilizar, mas valores muito altos também podem reduzir a precisão por excesso de generalização.
2. Aplique o KNN em outra base de dados (ex: Wine ou Breast Cancer).
    > O código feito com a base de dados Wine está [aqui](./knn_wine.py)
3. Compare o KNN com outros algoritmos supervisionados.
    > Nesta etapa, comparei o KNN com outros 2 algoritmos de classificação supervisionada utilizando o conjunto de dados Wine: Árvore de Decisão e Naive Bayes. No experimento realizado com o dataset, o algoritmo KNN apresentou a maior acurácia entre os três testados, superando tanto a Árvore de Decisão quanto o Naive Bayes. Isso indica que, nesse conjunto de dados específico, o KNN conseguiu classificar melhor os exemplos do conjunto de teste. Uma possível explicação para isso é que o dataset Wine possui atributos contínuos e bem comportados após a normalização, o que favorece algoritmos baseados em distância, como o KNN



---
Alunos: 
- Francisco Alexandro
- Gabriel da Silva Leal
