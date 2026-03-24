
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# 80/20 train-test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Constrained Decision Tree ---
dt_constrained = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=5,
    min_samples_split=10,
    random_state=42
)
dt_constrained.fit(X_train, y_train)
dt_preds = dt_constrained.predict(X_test)

# --- Neural Network ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = Sequential([
    Dense(32, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, verbose=0)
nn_preds = (model.predict(X_test_scaled) > 0.5).astype(int).flatten()

# --- Confusion Matrices ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Decision Tree confusion matrix
cm_dt = confusion_matrix(y_test, dt_preds)
disp_dt = ConfusionMatrixDisplay(confusion_matrix=cm_dt, display_labels=data.target_names)
disp_dt.plot(ax=axes[0], cmap='Blues')
axes[0].set_title("Constrained Decision Tree")

# Neural Network confusion matrix
cm_nn = confusion_matrix(y_test, nn_preds)
disp_nn = ConfusionMatrixDisplay(confusion_matrix=cm_nn, display_labels=data.target_names)
disp_nn.plot(ax=axes[1], cmap='Oranges')
axes[1].set_title("Neural Network")

plt.tight_layout()
plt.savefig("Q5_confusion_matrices.png", dpi=150)
plt.show()

print("Decision Tree Test Accuracy:", accuracy_score(y_test, dt_preds))
print("Neural Network Test Accuracy:", accuracy_score(y_test, nn_preds))

# Discussion:
# For this medical diagnosis task, I would prefer the Decision Tree because:
# - In healthcare, interpretability is critical. Doctors need to understand
#   WHY a model made a prediction, and decision trees provide clear,
#   human-readable rules.
#
# Decision Tree:
#   Advantage: Highly interpretable — the decision path can be visualized and
#   explained to non-technical stakeholders (e.g., doctors and patients).
#   Limitation: Limited capacity to capture complex, non-linear relationships
#   in the data, which may reduce performance on more complex datasets.
#
# Neural Network:
#   Advantage: Can learn complex non-linear patterns and often achieves higher
#   accuracy on large, high-dimensional datasets.
#   Limitation: Acts as a "black box" — it is difficult to interpret why a
#   specific prediction was made, which is a concern in high-stakes medical
#   applications.
