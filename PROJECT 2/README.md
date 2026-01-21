
# Breast Cancer Detection using Logistic Regression

A machine learning project that uses logistic regression to classify breast cancer tumors as malignant or benign based on tumor characteristics.

## Overview

This project implements a binary classification model using the Wisconsin Breast Cancer dataset from scikit-learn. The model achieves approximately 98% accuracy in predicting whether a breast tumor is malignant or benign.

## Dataset

- **Source**: Wisconsin Breast Cancer Dataset (sklearn.datasets)
- **Samples**: 569 instances
- **Features**: 30 numerical features describing tumor characteristics
  - Mean radius, texture, perimeter, area
  - Smoothness, compactness, concavity
  - Concave points, symmetry, fractal dimension
  - And their respective "worst" values
- **Target Classes**: 
  - Malignant (0)
  - Benign (1)

## Dependencies

```python
numpy
pandas
matplotlib
scikit-learn
```

Install dependencies using:
```bash
pip install numpy pandas matplotlib scikit-learn
```

## Project Structure

The notebook follows a standard machine learning workflow:

1. **Data Loading**: Load the breast cancer dataset from scikit-learn
2. **Data Exploration**: Examine features, target distribution, and dataset shape
3. **Train-Test Split**: Split data into 80% training and 20% testing sets (stratified)
4. **Feature Scaling**: Standardize features using StandardScaler (mean ≈ 0, std ≈ 1)
5. **Model Training**: Train a Logistic Regression classifier
6. **Evaluation**: Assess model performance using multiple metrics
7. **Visualization**: Create confusion matrix heatmap

## Model Performance

- **Accuracy**: 98.25%
- **Precision**: 
  - Malignant: 0.98
  - Benign: 0.99
- **Recall**: 
  - Malignant: 0.98
  - Benign: 0.99

### Confusion Matrix

```
              Predicted
           Malignant  Benign
Actual
Malignant     41        1
Benign         1       71
```

The model correctly classified 112 out of 114 test samples, with only 2 misclassifications.

## Key Implementation Details

### Feature Scaling
StandardScaler is applied to normalize features, which is crucial for logistic regression as features have different scales (e.g., area vs. smoothness).

### Logistic Regression
- Uses sigmoid function to compute probabilities
- Threshold: 0.5 (default)
- Max iterations: 1000
- Finds optimal weights for each feature to maximize classification accuracy

### Stratified Split
The train-test split uses `stratify=y` to maintain the same class ratio in both training and testing sets, ensuring representative evaluation.

## Usage

1. Open the notebook in Google Colab or Jupyter
2. Run all cells sequentially
3. The model will train and display:
   - Dataset statistics
   - Model accuracy
   - Classification report
   - Confusion matrix visualization

## Results Interpretation

The high accuracy (98.25%) and balanced precision/recall scores indicate that the model performs well for both classes. The confusion matrix shows:
- 41 true negatives (correctly identified malignant)
- 71 true positives (correctly identified benign)
- 1 false positive (malignant classified as benign)
- 1 false negative (benign classified as malignant)

## Future Improvements

- Try other classification algorithms (SVM, Random Forest, Neural Networks)
- Implement cross-validation for more robust evaluation
- Perform feature importance analysis
- Experiment with hyperparameter tuning
- Add ROC curve and AUC score analysis
- Implement ensemble methods for better performance

## Medical Disclaimer

This is an educational project for learning machine learning concepts. It should **NOT** be used for actual medical diagnosis. Always consult qualified healthcare professionals for medical decisions.

## License

This project uses the Wisconsin Breast Cancer Dataset, which is publicly available through scikit-learn.

## Author

Created as Project 2 - Machine Learning Classification Task

---

**Note**: The model demonstrates that machine learning can assist in medical diagnosis, but human expertise and multiple diagnostic methods are essential in clinical settings.
