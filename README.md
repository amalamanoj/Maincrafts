# House Price Prediction using Linear Regression

## Project Overview
This project predicts house prices using the California Housing Dataset and a Linear Regression model. The model is trained, evaluated, and saved for future use.

## Features
- Data preprocessing
- Exploratory data analysis
- Linear Regression model training
- House price prediction
- Model evaluation
- Data visualization
- Model saving

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Jupyter Notebook

## Dataset
California Housing Dataset (Scikit-learn)

## Results
- MAE: 0.5332
- RMSE: 0.7456
- R² Score: 0.5758

## Project Files
- `task1_ml_linear_regression.ipynb`
- `task1_ml_linear_regression.py`
- `house_price_model.pkl`
- `House_Price_Prediction_Report.pdf`

# Feature Engineering, Model Optimization & Performance Comparison

## 📌 Project Overview

This project was completed as part of the Artificial Intelligence & Machine Learning Internship (Task 2). The objective is to build and compare multiple machine learning regression models for predicting California housing prices using the California Housing Dataset.

The project includes data preprocessing, feature scaling, model training, evaluation, and performance comparison.

---

## 🎯 Objectives

- Load and explore the California Housing Dataset.
- Perform data preprocessing and feature scaling.
- Train multiple regression models.
- Compare model performance using evaluation metrics.
- Visualize prediction results and feature importance.

---

## 📂 Dataset

- **Dataset:** California Housing Dataset
- **Source:** Scikit-learn
- **Target Variable:** Median House Value

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- VS Code

---

## 🤖 Machine Learning Models

- Linear Regression
- Ridge Regression
- Decision Tree Regressor

---

## 📊 Evaluation Metrics

- Root Mean Squared Error (RMSE)
- R² Score

---

## 📈 Results

| Model | RMSE | R² Score |
|-------|------|----------|
| Linear Regression | 0.7456 | 0.5758 |
| Ridge Regression | 0.7456 | 0.5758 |
| Decision Tree Regressor | 0.7030 | 0.6228 |

**Best Model:** Decision Tree Regressor

---

## 📁 Project Files

```
Task2_California_Housing/
│── task2.py
│── README.md
│── Task2_Report.pdf
│── Task2_Report.docx
```

---

## ▶️ How to Run

1. Install the required libraries:

```bash
pip install pandas numpy matplotlib scikit-learn
```

2. Run the project:

```bash
python task2.py
```

---

## 📌 Conclusion

The project successfully compared three regression algorithms for predicting California housing prices. Based on the evaluation metrics, the Decision Tree Regressor achieved the best performance among the tested models.

--

# AI & ML Task 3: Model Validation, Overfitting Control & Hyperparameter Tuning

## Project Overview

This project demonstrates the implementation of model validation techniques, overfitting control, and hyperparameter tuning using the Decision Tree Regression algorithm. The California Housing dataset from scikit-learn is used to build and evaluate a machine learning model.

The project compares the performance of a baseline Decision Tree model with an optimized model obtained using GridSearchCV.

---

## Objectives

- Load and explore the California Housing dataset.
- Split the dataset into training and testing sets.
- Train a baseline Decision Tree Regression model.
- Evaluate the model using RMSE and R² Score.
- Perform 5-Fold Cross-Validation.
- Tune hyperparameters using GridSearchCV.
- Compare the baseline and tuned models.

---

## Technologies Used

- Python 3.x
- NumPy
- Pandas
- Scikit-learn
- VS Code / Jupyter Notebook

---

## Dataset

**Dataset:** California Housing Dataset

- Source: Scikit-learn
- Samples: 20,640
- Features: 8
- Target: Median House Value

---

## Installation

Install the required libraries:

```bash
pip install numpy pandas scikit-learn
```

---

## How to Run

1. Open the project folder in VS Code.
2. Save the program as `task3.py`.
3. Open the terminal.
4. Run the following command:

```bash
python task3.py
```

---

## Project Workflow

1. Load the California Housing dataset.
2. Split the dataset into training and testing sets.
3. Train a baseline Decision Tree Regression model.
4. Evaluate the model using RMSE and R² Score.
5. Perform 5-Fold Cross-Validation.
6. Tune the model using GridSearchCV.
7. Evaluate the tuned model.
8. Compare the baseline and tuned models.
9. Display the final results.

---

## Model Evaluation Metrics

- Root Mean Squared Error (RMSE)
- R² Score
- Cross-Validation Score

---

## Output Summary

### Baseline Model

- RMSE: **0.7030**
- R² Score: **0.6228**

### Cross Validation

- Average R² Score: **0.6070**

### Best Hyperparameters

```text
max_depth = 10
min_samples_leaf = 4
min_samples_split = 2
```

### Tuned Model

- RMSE: **0.6391**
- R² Score: **0.6883**

---

## Results

The tuned Decision Tree model achieved:

- Lower RMSE
- Higher R² Score
- Better generalization through Cross-Validation
- Reduced overfitting using optimized hyperparameters

Therefore, the tuned model performs better than the baseline model.

---

## Learning Outcomes

After completing this project, I learned:

- Model validation techniques
- Train-Test Split
- Cross-Validation
- Overfitting and Underfitting
- Hyperparameter Tuning
- GridSearchCV
- Decision Tree Regression
- Model performance evaluation using RMSE and R² Score

---

## Future Improvements

- Apply Random Forest Regression.
- Apply XGBoost Regression.
- Compare multiple regression algorithms.
- Perform feature engineering for better accuracy.

---

## License

This project was developed for educational and internship purposes.

## Author
**Amala Manoj**  
B.Tech Computer Science and Engineering








