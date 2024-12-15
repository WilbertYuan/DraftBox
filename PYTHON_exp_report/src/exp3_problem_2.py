import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
# from scipy.stats import mode

def distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def classify_one_example(x, centers):
    min_i = 0
    min_dist = float('inf')
    for i in range(len(centers)):
        center = centers[i]
        dist = distance(x, center)
        if dist < min_dist:
            min_i = i
            min_dist = dist
    return min_i, min_dist

def re_classify_examples(X, example_bags, centers):
    for example_bag in example_bags:
        example_bag.clear()
    new_cost = 0
    for x in X:
        index, dist = classify_one_example(x, centers)
        example_bags[index].append(x)
        new_cost += dist
    return new_cost / len(X)

def cal_centers(example_bags, centers):
    for i in range(len(example_bags)):
        if len(example_bags[i]) > 0:
            centers[i] = np.mean(example_bags[i], axis=0)

def kmeans(X, K, max_iter=5000, tol=1e-20):
    centers = np.array([X[i] for i in range(K)])
    example_bags = [[] for _ in range(K)]
    cost = float('inf')

    for i in range(max_iter):
        new_cost = re_classify_examples(X, example_bags, centers)
        cal_centers(example_bags, centers)
        
        if i % 10 == 0:
            print(f"Iteration {i + 1}, Cost: {new_cost:.4f}")
        if np.abs(new_cost - cost) < tol:
            print(f"Stopped at iteration {i + 1}")
            break
        cost = new_cost

    return centers, example_bags


def accuracy(labels_true, labels_pred):
    return np.sum(labels_true == labels_pred) / len(labels_true)

data = np.loadtxt("data/number_data/arab_digits_training.txt", delimiter="\t")
X = data[:, 1:] 
y = data[:, 0]

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

K = 10
centers, example_bags = kmeans(X_pca, K)
labels_pred = np.array([np.argmin([distance(x, center) for center in centers]) for x in X_pca])

acc = accuracy(y, labels_pred)
print(f"Accuracy: {acc:.4f}")

plt.figure(figsize=(8, 6))


for i in range(K):
    example_bags[i] = np.array(example_bags[i])
    plt.scatter(example_bags[i][:, 0], example_bags[i][:, 1], label=f'Cluster {i}', s=5)

centers = np.array(centers)
plt.scatter(centers[:, 0], centers[:, 1], c='black', marker='+', s=200, label='Centroids')

plt.title('K-Means Clustering on PCA-reduced Data', fontsize=16)
plt.xlabel('Principal Component 1', fontsize=12)
plt.ylabel('Principal Component 2', fontsize=12)
plt.legend()
plt.grid(True)
plt.savefig("img/problem3-2.png")
plt.show()
