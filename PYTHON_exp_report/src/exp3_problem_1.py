from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train_data = pd.read_csv('data/experiment_2/题目2训练数据.txt', header=None, delim_whitespace=True)
test_data = pd.read_csv('data/experiment_2/题目2测试数据.txt',header=None,delim_whitespace=True)

X = train_data.iloc[:, :-1]
y = train_data.iloc[:, -1]
X_test = test_data.iloc[:,:]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = KNeighborsClassifier(n_neighbors=2,algorithm='brute',weights='distance')

model.fit(X_train, y_train)

y_val_pred = model.predict(X_val)

accuracy = accuracy_score(y_val, y_val_pred)
precision = precision_score(y_val, y_val_pred)
recall = recall_score(y_val, y_val_pred)

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')

y_test = model.predict(X_test).T
np.savetxt('result/prediction3-1.txt', y_test, fmt='%.2f')

print(X_test)
print(y_test)
d = np.concatenate((X_test, y_test.reshape(-1, 1)), axis=1)
d1 = np.array([i for i in d if i[-1] == 1])
d0 = np.array([i for i in d if i[-1] == 0])

xfeature, yfeature = 0, 13
plt.figure(figsize=(8, 6))
plt.scatter(d1[:,xfeature],d1[:,yfeature],label='y = 1',alpha=0.5,s=1)
plt.scatter(d0[:,xfeature],d0[:,yfeature],label='y = 0',alpha=0.5,s=1)
plt.xlabel(f'feature {xfeature}')
plt.ylabel(f'feature {yfeature}')
plt.legend()
plt.title(f'visualization feature {xfeature} & feature {yfeature}')
plt.savefig("img/problem3-1.png")
plt.show()