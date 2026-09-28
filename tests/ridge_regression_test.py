from si.io.csv_file import read_csv
from si.model_selection.split import train_test_split
from si.models.linear_regression import RidgeRegression

dataset = read_csv("datasets/cpu/cpu.csv", sep=",", features=True, label=True)
print("Dataset:", dataset.X.shape)

train, test = train_test_split(dataset, test_size=0.2, random_state=42)


model = RidgeRegression(l2_penalty=1.0, alpha=0.01, max_iter=2000, patience=100, scale=True)
model.fit(train)

print("Score treino (MSE):", model.score(train))
print("Score teste  (MSE):", model.score(test))


print("Custo final do treino:", list(model.cost_history.values())[-1])
print("Custo no teste:", model.cost(test.y, model.predict(test)))
print("Iterações feitas:", len(model.cost_history))

