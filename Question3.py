

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# 80/20 train-test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train a constrained Decision Tree (max_depth=5, min_samples_split=10)
dt_constrained = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=5,
    min_samples_split=10,
    random_state=42
)
dt_constrained.fit(X_train, y_train)

# Report training and test accuracy
train_acc = accuracy_score(y_train, dt_constrained.predict(X_train))
test_acc = accuracy_score(y_test, dt_constrained.predict(X_test))

print(f"Constrained DT Training Accuracy: {train_acc:.4f}")
print(f"Constrained DT Test Accuracy:     {test_acc:.4f}")

# Display top 5 most important features
importances = dt_constrained.feature_importances_
feature_names = data.feature_names
top5_idx = np.argsort(importances)[::-1][:5]

print("\nTop 5 Most Important Features:")
for rank, idx in enumerate(top5_idx, 1):
    print(f"  {rank}. {feature_names[idx]}: {importances[idx]:.4f}")

# Discussion:
# Controlling model complexity (e.g., limiting max_depth or increasing
# min_samples_split) helps prevent overfitting by stopping the tree from
# growing too deep and memorizing noise in the training data. A constrained
# tree is simpler, which typically leads to better generalization on unseen
# data — even if training accuracy decreases slightly.
#
# Feature importance in decision trees is based on how much each feature
# contributes to reducing impurity (entropy) across all splits. This makes
# decision trees highly interpretable: we can clearly see which features
# the model relies on most for its predictions, which is especially valuable
# in medical applications where understanding the reasoning behind a
# diagnosis is critical.
