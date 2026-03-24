
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# 80/20 train-test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardize the input features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build a neural network with one hidden layer and sigmoid output
model = Sequential([
    Dense(32, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    X_train_scaled, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# Report training and test accuracy
train_loss, train_acc = model.evaluate(X_train_scaled, y_train, verbose=0)
test_loss, test_acc = model.evaluate(X_test_scaled, y_test, verbose=0)

print(f"\nNeural Network Training Accuracy: {train_acc:.4f}")
print(f"Neural Network Test Accuracy:     {test_acc:.4f}")

# Discussion:
# Feature scaling (standardization) is necessary for neural networks because:
# - Neural networks use gradient-based optimization. If features have very
#   different scales, gradients will be disproportionately influenced by
#   larger-scaled features, leading to slow or unstable convergence.
# - Standardizing features to have mean=0 and std=1 ensures that all features
#   contribute equally during training and the optimizer converges faster.
#
# An epoch represents one complete pass through the entire training dataset.
# During each epoch, the model sees every training sample once, computes
# the loss, and updates the weights via backpropagation. Training for multiple
# epochs allows the model to iteratively refine its weights and improve
# its predictions.
