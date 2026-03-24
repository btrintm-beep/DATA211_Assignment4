# Data Science Assignment 4 — Decision Trees, Neural Networks & CNNs

## Files

| File | Question | Topic |
|------|----------|-------|
| `Q1_Dataset_Exploration.py` | Q1 (5 pts) | Dataset exploration, class balance |
| `Q2_Decision_Tree_Entropy.py` | Q2 (10 pts) | Decision Tree with entropy criterion |
| `Q3_Tree_Complexity.py` | Q3 (10 pts) | Constrained Decision Tree, feature importance |
| `Q4_Neural_Network.py` | Q4 (10 pts) | Neural Network for binary classification |
| `Q5_Model_Comparison.py` | Q5 (5 pts) | Confusion matrices, model comparison |
| `Q6_CNN_Fashion_MNIST.py` | Q6 (10 pts) | CNN on Fashion MNIST |
| `Q7_CNN_Error_Analysis.py` | Q7 (10 pts) | Misclassification analysis |

## Datasets

- **Q1–Q5**: Breast Cancer Wisconsin (Diagnostic) via `sklearn.datasets`
- **Q6–Q7**: Fashion MNIST via `tensorflow.keras.datasets`

## Requirements

```
scikit-learn
tensorflow
matplotlib
numpy
```

## How to Run

Run each file individually in order (Q6 must run before Q7 since Q7 loads the saved model):

```bash
python Q1_Dataset_Exploration.py
python Q2_Decision_Tree_Entropy.py
python Q3_Tree_Complexity.py
python Q4_Neural_Network.py
python Q5_Model_Comparison.py
python Q6_CNN_Fashion_MNIST.py
python Q7_CNN_Error_Analysis.py
```

**Note:** Q6 saves the trained CNN model as `fashion_cnn_model.h5`, which Q7 loads for error analysis.
