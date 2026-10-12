from si.io.csv_file import read_csv
from si.model_selection.split import train_test_split
from si.models.logistic_regression import LogisticRegression

# 1. carregar o dataset
dataset = read_csv("datasets/breast-bin.csv", sep=",", features=True, label=True)
print("Dataset:", dataset.X.shape)

# 2. dividir em treino e teste
train, test = train_test_split(dataset, test_size=0.2, random_state=42)

# 3. treinar o modelo
model = LogisticRegression(l2_penalty=1.0, alpha=0.01, max_iter=2000, patience=100, scale=True)
model.fit(train)

print("Score treino (accuracy):", model.score(train))
print("Score teste  (accuracy):", model.score(test))

print("Custo final do treino:", list(model.cost_history.values())[-1])
print("Custo no teste:", model.cost(test.y, model.predict(test)))
print("Iterações feitas:", len(model.cost_history))