# House Price Prediction using Linear Regression

## Project Overview

This project predicts house prices using a Linear Regression Machine Learning model.

The model uses important house features such as:

- Living Area (GrLivArea)
- Number of Bedrooms (BedroomAbvGr)
- Number of Bathrooms (FullBath)

The target variable is SalePrice.

## Objective

The main objective of this project is to build a Machine Learning model that can predict house prices based on selected house features.

## Dataset

The dataset used in this project is the House Prices dataset from Kaggle.

The dataset contains information about residential houses and their sale prices.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

## Machine Learning Algorithm

### Linear Regression

Linear Regression is a supervised machine learning algorithm used to predict a continuous numerical value.

In this project:

Input Features:
- GrLivArea
- BedroomAbvGr
- FullBath

Target:
- SalePrice

## Project Workflow

1. Load the dataset.
2. Select important features.
3. Check missing values.
4. Split the dataset into training and testing data.
5. Train a Linear Regression model.
6. Predict house prices.
7. Evaluate the model.
8. Visualize actual vs predicted prices.
9. Take user input and predict the price of a new house.

## Model Evaluation

The model is evaluated using:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- R² Score

The model achieved an R² Score of approximately 0.63 using the selected features.

## User Input

The program allows the user to enter:

- House living area in square feet
- Number of bedrooms
- Number of bathrooms

The trained model then predicts the estimated house price.

## How to Run

### 1. Install Required Libraries

```bash
pip install pandas numpy matplotlib scikit-learn