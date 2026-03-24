
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.datasets import fashion_mnist
import numpy as np

# Load the Fashion MNIST dataset
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# Class names for reference
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape:     {X_test.shape}")

# Normalize pixel values to [0, 1]
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Reshape to include channel dimension (28, 28) -> (28, 28, 1)
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Train the model for at least 15 epochs
history = model.fit(
    X_train, y_train,
    epochs=15,
    batch_size=64,
    validation_split=0.1,
    verbose=1
)

# Report test accuracy
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_acc:.4f}")

# Save the model for use in Q7
model.save("fashion_cnn_model.h5")

# Discussion:
# CNNs are generally preferred over fully connected networks for image data
# because:
# - CNNs use convolutional filters that exploit the spatial structure of images.
#   They detect local patterns (edges, textures, shapes) regardless of their
#   position in the image (translation invariance).
# - Fully connected networks treat each pixel as an independent feature, ignoring
#   spatial relationships and requiring far more parameters, which leads to
#   overfitting and higher computational cost.
# - CNNs use weight sharing and pooling to dramatically reduce the number of
#   parameters while preserving important spatial features.
#
# The convolution layer in this task is learning to detect low-level spatial
# patterns such as edges, textures, and shapes in the clothing images. The first
# convolutional layer typically learns simple features like horizontal/vertical
# edges, while deeper layers combine these into more complex patterns like
# collars, sleeves, or shoe outlines that help distinguish between categories.
