import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

train_data = pd.read_csv('data/experiment_2/题目2训练数据.txt', header=None, delim_whitespace=True)
test_data = pd.read_csv('data/experiment_2/题目2测试数据.txt', header=None, delim_whitespace=True)

X_train = train_data.iloc[:, :-1]
y_train = train_data.iloc[:, -1]
X_test = test_data.iloc[:,:]
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

model = LogisticRegression(
    penalty='l2',
    C=1.0,
    solver='saga',
    max_iter=500,
    class_weight='balanced',
    random_state=42
)

model.fit(X_train, y_train)

y_train_prob = model.predict_proba(X_train)[:, 1]
y_val_prob = model.predict_proba(X_val)[:, 1]

train_auc = roc_auc_score(y_train, y_train_prob)
val_auc = roc_auc_score(y_val, y_val_prob)

print(f'Training AUC: {train_auc:.4f}')
print(f'Validation AUC: {val_auc:.4f}')


y_test_prob = model.predict_proba(X_test)[:, 1]
y_test = [1 if i > 0.5 else 0 for i in y_test_prob]
np.savetxt('result/prediction2-2.txt', y_test, fmt='%.2f')

plt.figure(figsize=(8, 6))
plt.scatter(range(len(y_train_prob)), y_train_prob,label='Training Prediction Probabilities', alpha=0.5,s = 0.5)
plt.scatter(range(len(y_val_prob)), y_val_prob,label='Validation Prediction Probabilities', alpha=0.5, s = 0.5)
plt.title(f'Prediction Probabilities (Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f})')
plt.xlabel('Samples')
plt.ylabel('Prediction Probability')
plt.legend()
plt.savefig('img/problem_2_visual.png')
plt.show()
