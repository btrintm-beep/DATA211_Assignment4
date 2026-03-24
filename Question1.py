
from sklearn.datasets import load_breast_cancer
import numpy as np

# Load dataset
data = load_breast_cancer()

# Construct feature matrix X and target vector y
X = data.data
y = data.target

# Report the shape of X and y
print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

# Report the number of samples belonging to each class
unique, counts = np.unique(y, return_counts=True)
for label, count in zip(unique, counts):
    print(f"Class {label} ({data.target_names[label]}): {count} samples")

# Discussion:
# The dataset is slightly imbalanced — there are more benign (class 1) samples
# than malignant (class 0) samples. However, the imbalance is not extreme.
#
# Class balance is an important consideration for classification models because:
# - If one class dominates, the model may become biased toward predicting the
#   majority class, leading to high overall accuracy but poor recall for the
#   minority class.
# - In a medical context like cancer diagnosis, failing to detect malignant
#   cases (false negatives) can have serious consequences, so it is critical
#   that the model performs well on both classes.
# - Techniques like stratified splitting, class weighting, or resampling can
#   help address imbalance issues.
