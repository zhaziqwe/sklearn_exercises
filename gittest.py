import numpy as np
from collections import Counter

class Node:
    def __init__(self, feature=None, threshold=None, label=None, left=None, right=None):
        self.feature = feature  # 划分特征的索引
        self.threshold = threshold  # 划分特征的阈值
        self.label = label  # 叶节点的类别
        self.left = left  # 左子节点
        self.right = right  # 右子节点

def gini_index(labels):
    total_samples = len(labels)
    counts = Counter(labels)
    gini = 1.0
    for count in counts.values():
        prob = count / total_samples
        gini -= prob ** 2
    return gini

def build_tree(X, y):
    if len(set(y)) == 1:
        # 如果所有样本属于同一类别，创建叶节点
        return Node(label=y[0])
    else:
        best_gini = float('inf')
        best_feature = None
        best_threshold = None
        n_features = X.shape[1]
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_indices = np.where(X[:, feature] <= threshold)[0]
                right_indices = np.where(X[:, feature] > threshold)[0]
                left_gini = gini_index(y[left_indices])
                right_gini = gini_index(y[right_indices])
                gini = (len(left_indices) * left_gini + len(right_indices) * right_gini) / len(y)
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_threshold = threshold

        left_indices = np.where(X[:, best_feature] <= best_threshold)[0]
        right_indices = np.where(X[:, best_feature] > best_threshold)[0]
        left_subtree = build_tree(X[left_indices], y[left_indices])
        right_subtree = build_tree(X[right_indices], y[right_indices])
        return Node(feature=best_feature, threshold=best_threshold, left=left_subtree, right=right_subtree)
    
def predict(node, sample):
    if node.label is not None:
        return node.label
    else:
        if sample[node.feature] <= node.threshold:
            return predict(node.left, sample)
        else:
            return predict(node.right, sample)
        
X = np.array([[2.9, 6.7], [1.7, 5.4], [7.5, 3.2], [6.3, 0.9], [4.1, 2.7]])
y = np.array([0, 0, 1, 1, 1])

tree = build_tree(X, y)
sample = np.array([3.6, 5.1])
prediction = predict(tree, sample)
print("answer:", prediction)