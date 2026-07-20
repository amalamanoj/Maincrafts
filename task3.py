# ==========================================================
# AI & ML Task 3
# Model Validation, Overfitting Control & Hyperparameter Tuning
# ==========================================================

# Import libraries
import pandas as pd
import numpy as np

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ----------------------------------------------------------
# Load Dataset
# ----------------------------------------------------------

housing = fetch_california_housing()

X = housing.data
y = housing.target

print("Dataset Loaded Successfully")
print("Features:", housing.feature_names)
print("Number of Samples:", len(X))

# ----------------------------------------------------------
# Train-Test Split
# ----------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# ----------------------------------------------------------
# Baseline Model
# ----------------------------------------------------------

model = DecisionTreeRegressor(random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n========== Baseline Model ==========")
print("RMSE:", rmse)
print("R² Score:", r2)

# ----------------------------------------------------------
# Cross Validation
# ----------------------------------------------------------

scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=5,
    scoring="r2"
)

print("\n========== Cross Validation ==========")
print("Cross Validation Scores:", scores)
print("Average Score:", scores.mean())

# ----------------------------------------------------------
# Hyperparameter Tuning
# ----------------------------------------------------------

parameters = {
    "max_depth": [3,5,7,10,None],
    "min_samples_split": [2,5,10],
    "min_samples_leaf": [1,2,4]
}

grid = GridSearchCV(
    DecisionTreeRegressor(random_state=42),
    parameters,
    cv=5,
    scoring="r2",
    n_jobs=-1
)

grid.fit(X_train, y_train)

print("\n========== Best Parameters ==========")
print(grid.best_params_)

# ----------------------------------------------------------
# Best Model
# ----------------------------------------------------------

best_model = grid.best_estimator_

prediction = best_model.predict(X_test)

rmse_best = np.sqrt(mean_squared_error(y_test, prediction))
r2_best = r2_score(y_test, prediction)

print("\n========== Tuned Model ==========")
print("RMSE:", rmse_best)
print("R² Score:", r2_best)

# ----------------------------------------------------------
# Model Comparison
# ----------------------------------------------------------

print("\n========== Comparison ==========")

print("Baseline RMSE :", rmse)
print("Tuned RMSE    :", rmse_best)

print("Baseline R²   :", r2)
print("Tuned R²      :", r2_best)

if r2_best > r2:
    print("\nThe tuned model performs better than the baseline model.")
else:
    print("\nThe baseline model performs better.")

print("\nTask 3 Completed Successfully!")