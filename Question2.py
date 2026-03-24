

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# 80/20 train-test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train a Decision Tree classifier using entropy
dt_model = DecisionTreeClassifier(criterion='entropy', random_state=42)
dt_model.fit(X_train, y_train)

# Report training and test accuracy
train_acc = accuracy_score(y_train, dt_model.predict(X_train))
test_acc = accuracy_score(y_test, dt_model.predict(X_test))

print(f"Training Accuracy: {train_acc:.4f}")
print(f"Test Accuracy:     {test_acc:.4f}")

# Discussion:
# Entropy, in the context of decision trees, measures the level of uncertainty
# or impurity in a set of samples. An entropy of 0 means all samples belong to
# one class (pure node), while maximum entropy occurs when classes are equally
# distributed. The decision tree selects splits that maximize information gain,
# which is the reduction in entropy after splitting on a feature.
#
# If the training accuracy is significantly higher than the test accuracy
# (e.g., training accuracy near 1.0 while test accuracy is noticeably lower),
# this suggests overfitting — the model has memorized the training data
# including noise and does not generalize well to unseen data.
# If both accuracies are close, the model generalizes reasonably well.
