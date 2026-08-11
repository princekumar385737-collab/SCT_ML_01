import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# --------------------------------
# 1. Load Dataset
# --------------------------------

data = pd.read_csv("train.csv")

print("Dataset loaded successfully!")
print("Dataset Shape:", data.shape)


# --------------------------------
# 2. Select Required Features
# --------------------------------

X = data[['GrLivArea', 'BedroomAbvGr', 'FullBath']]
y = data['SalePrice']


# --------------------------------
# 3. Check Missing Values
# --------------------------------

print("\nMissing Values:")
print(X.isnull().sum())


# --------------------------------
# 4. Split Dataset
# --------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)


# --------------------------------
# 5. Create and Train Model
# --------------------------------

model = LinearRegression()

model.fit(X_train, y_train)

print("\nLinear Regression model trained successfully!")


# --------------------------------
# 6. Make Predictions
# --------------------------------

y_pred = model.predict(X_test)


# --------------------------------
# 7. Model Evaluation
# --------------------------------

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("R2 Score:", r2)


# --------------------------------
# 8. Actual vs Predicted Graph
# --------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual House Price")
plt.ylabel("Predicted House Price")
plt.title("Actual vs Predicted House Prices")

plt.show()


# --------------------------------
# 9. User Input for New House
# --------------------------------

print("\n--------------------------------")
print("House Price Prediction")
print("--------------------------------")

grlivarea = float(input("Enter house living area (sq ft): "))
bedrooms = int(input("Enter number of bedrooms: "))
bathrooms = int(input("Enter number of bathrooms: "))


new_house = pd.DataFrame({
    'GrLivArea': [grlivarea],
    'BedroomAbvGr': [bedrooms],
    'FullBath': [bathrooms]
})


# --------------------------------
# 10. Predict New House Price
# --------------------------------

predicted_price = model.predict(new_house)[0]

print("\nHouse Details:")
print("Living Area:", grlivarea, "sq ft")
print("Bedrooms:", bedrooms)
print("Bathrooms:", bathrooms)

print("\nPredicted House Price:", round(predicted_price, 2))