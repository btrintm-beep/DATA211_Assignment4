
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import fashion_mnist
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

# Class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Load dataset
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# Normalize and reshape
X_test_processed = X_test.astype('float32') / 255.0
X_test_processed = X_test_processed.reshape(-1, 28, 28, 1)

# Load trained model from Q6 (or retrain if needed)
try:
    model = load_model("fashion_cnn_model.h5")
    print("Model loaded from file.")
except:
    print("Model file not found — please run Q6 first to train and save the model.")
    exit()

# Generate predictions on the test set
y_pred_probs = model.predict(X_test_processed)
y_pred = np.argmax(y_pred_probs, axis=1)

# --- Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(ax=ax, cmap='Blues', xticks_rotation=45)
ax.set_title("Fashion MNIST CNN — Confusion Matrix")
plt.tight_layout()
plt.savefig("Q7_confusion_matrix.png", dpi=150)
plt.show()

# --- Identify and visualize misclassified images ---
misclassified_idx = np.where(y_pred != y_test)[0]
print(f"\nTotal misclassified images: {len(misclassified_idx)} out of {len(y_test)}")

# Show at least 3 misclassified images
num_to_show = 5
fig, axes = plt.subplots(1, num_to_show, figsize=(15, 3))
fig.suptitle("Misclassified Images", fontsize=14)

for i in range(num_to_show):
    idx = misclassified_idx[i]
    axes[i].imshow(X_test[idx], cmap='gray')
    axes[i].set_title(
        f"True: {class_names[y_test[idx]]}\nPred: {class_names[y_pred[idx]]}",
        fontsize=9, color='red'
    )
    axes[i].axis('off')

plt.tight_layout()
plt.savefig("Q7_misclassified_images.png", dpi=150)
plt.show()

# Discussion:
# One pattern observed in the misclassifications:
# The model frequently confuses visually similar categories, such as
# Shirt vs T-shirt/top, and Pullover vs Coat. These items share similar
# shapes and textures in 28x28 grayscale images, making them hard to
# distinguish even for humans at this low resolution.
#
# One realistic method to improve CNN performance:
# Data augmentation — applying random transformations during training such as
# rotation, horizontal flipping, zooming, and shifting. This increases the
# effective diversity of the training data and helps the model generalize
# better to variations in clothing appearance without needing additional
# labeled data.
