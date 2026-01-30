# Python Data Science & Machine Learning Basics

This repository contains Jupyter notebooks demonstrating fundamental concepts in Python programming, data visualization, and machine learning.

## 📚 Contents

### Python Fundamentals
- Basic Python syntax and operations
- Data types (lists, tuples, sets, dictionaries)
- String manipulation and operations
- Control flow (conditionals, loops)
- Error handling

### Data Visualization
- **Matplotlib**: Creating bar charts and visualizations
- Sample visualization: Average marks per subject

### Machine Learning Examples

#### 1. Linear Regression
- **Dataset**: Hours Studied vs Marks
- **Objective**: Predict student marks based on study hours
- **Key Concepts**:
  - Training a linear regression model
  - Understanding the relationship between features and target
  - Making predictions on new data
- **Visualization**: Scatter plot with fitted regression line

#### 2. Classification (Iris Dataset)
- **Dataset**: Iris flowers (2 features for visualization)
- **Objective**: Classify flower species
- **Algorithm**: K-Nearest Neighbors (KNN)
- **Features Used**: Sepal length and sepal width
- **Accuracy**: ~80% on test set

## 🚀 Getting Started

### Prerequisites
```python
numpy
pandas
matplotlib
scikit-learn
tensorflow
```

### Installation
```bash
pip install numpy pandas matplotlib scikit-learn tensorflow
```

## 📊 Key Visualizations

1. **Bar Chart**: Average marks across different subjects
2. **Scatter Plot**: Linear relationship between study hours and marks
3. **Classification Plot**: Iris dataset visualization by species

## 🎯 Learning Outcomes

- Understand basic Python programming concepts
- Create visualizations using Matplotlib
- Implement supervised learning algorithms:
  - Linear Regression for continuous predictions
  - KNN for classification tasks
- Evaluate model performance
- Understand the difference between regression and classification

## 📖 Usage

Open the notebook in Google Colab or Jupyter:

```python
# Example: Making predictions with the trained model
my_hours = np.array([[5]])  # Study 5 hours
predicted_marks = model.predict(my_hours)
print(f"Predicted marks: {predicted_marks[0]:.2f}")
```

## 🔍 Real-World Applications

- **Linear Regression**: Salary prediction, price forecasting, demand estimation
- **Classification**: Customer segmentation, image recognition, disease diagnosis

## 📝 Notes

- The linear regression model formula: `marks ≈ m * hours + b`
- KNN classifier uses 5 nearest neighbors for classification
- Data is split 80/20 for training and testing

## 🤝 Contributing

Feel free to fork this repository and submit pull requests for improvements.

## 📄 License

This project is for educational purposes.

---

**Author**: Data Science Learning Journey  
**Last Updated**: 2026-01-19
