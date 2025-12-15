# Salary Prediction using Machine Learning

This project focuses on predicting annual salary based on demographic and work-related features using machine learning regression models.

The goal is to explore how factors such as age, education level, and weekly working hours influence salary, and to compare the performance of different machine learning algorithms.

## Dataset
The project uses the **Adult dataset** from the UCI Machine Learning Repository, loaded directly via **OpenML** using `scikit-learn`.
No local CSV files are used.

## Features
- Age
- Education level (encoded using education-num)
- Hours worked per week

## Target
- Estimated annual salary (generated using a realistic economic formula)

## Machine Learning Models
Three regression models are trained and compared:
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

Model performance is evaluated using:
- Mean Squared Error (MSE)
- R² score (coefficient of determination)

## Visualizations
- Scatter plot: Age vs Salary with regression line
- Bar chart comparing model performance (R²)

## User Input
The project allows interactive input from the user:
- Age
- Education level (human-readable selection)
- Weekly working hours

Based on the best-performing model, the system predicts the expected annual salary.

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

## Academic Context
This project was developed as part of the course **"Introduction to Machine Learning"** at **Technical University of Varna**.

## 👤 Author
**Victoria Kostadinova**
