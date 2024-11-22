import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

np.random.seed(42)
data = np.loadtxt("data/experiment_2/题目1数据.txt")
X, y = data[:, :-1], data[:, -1]

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test = X[:350], X[350:414]
y_train, y_test = y[:350], y[350:414]

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

num_features = X.shape[1]
fig, axes = plt.subplots(num_features, 1, figsize=(10, 5 * num_features))

rss = np.sum((y_test - y_pred) ** 2)
tss = np.sum((y_test - np.mean(y_test)) ** 2)
r2 = 1 - rss / tss

print("Manual R^2 and analysis results:")
print(f"R^2: {r2:.4f}")
print(f"RSS: {rss:.4f}")

for i in range(num_features):
    ax = axes[i] if num_features > 1 else axes
    ax.scatter(X_train[:, i], y_train, color='blue', label='Train', alpha=0.7,s=5)
    ax.scatter(X_test[:, i], y_test, color='red', label='Test', alpha=0.7,s=5)
    
    x_vals = np.linspace(X[:, i].min(), X[:, i].max(), 100).reshape(-1, 1)
    x_vals_all = np.zeros((100, num_features))
    x_vals_all[:, i] = x_vals[:, 0]
    y_vals = model.predict(x_vals_all)
    
    ax.plot(x_vals, y_vals, color='green', label='Fit line')
    ax.set_title(f"Feature {i + 1} vs Price")
    ax.set_xlabel(f"Feature {i + 1}")
    ax.set_ylabel("Price")
    y_min = min(y_train.min(), y_test.min(), y_vals.min())
    y_max = max(y_train.max(), y_test.max(), y_vals.max())
    ax.set_ylim(y_min, y_max)
    ax.legend()

plt.savefig("img/features.png")

plt.tight_layout()
plt.show()