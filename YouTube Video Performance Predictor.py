# ----------------------------------------------------
# STEP 1: Import required libraries
# ----------------------------------------------------

import numpy as np          # Numerical calculations
import pandas as pd         # Data handling (tables)
import matplotlib.pyplot as plt  # Graphs

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# ----------------------------------------------------
# STEP 2: Create the dataset
# ----------------------------------------------------

data = {
    'CTR_percent': [3.2, 5.8, 2.1, 7.4, 4.5, 6.2, 1.8, 5.1, 3.9, 8.5,
                    4.2, 2.8, 6.8, 3.5, 7.1, 2.4, 5.5, 4.8, 6.5, 3.1,
                    7.8, 2.6, 5.3, 4.1, 6.9, 3.7, 8.1, 2.2, 5.9, 4.6],

    'Total_Views': [12000, 28000, 8500, 42000, 19000, 33000, 7000, 24000, 16000, 51000,
                    18000, 11000, 38000, 14500, 44000, 9500, 26000, 21000, 35000, 13000,
                    47000, 10000, 25000, 17500, 40000, 15000, 49000, 8000, 31000, 20000]
}

# Convert dictionary into DataFrame
df = pd.DataFrame(data)

print("Dataset Preview:")
print(df.head())


# ----------------------------------------------------
# STEP 3: Separate input (X) and output (y)
# ----------------------------------------------------

X = df[['CTR_percent']]      # Input feature
y = df['Total_Views']        # Output value


# ----------------------------------------------------
# STEP 4: Split data into training and testing
# ----------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ----------------------------------------------------
# STEP 5: Create and train the model
# ----------------------------------------------------

model = LinearRegression()
model.fit(X_train, y_train)


# ----------------------------------------------------
# STEP 6: Display the learned equation
# ----------------------------------------------------

x1 = df['CTR_percent'].mean()
y1 = df['Total_Views'].mean()

numerator = ((df['CTR_percent'] - x1) * (df['Total_Views'] - y1)).sum()
denominator = ((df['CTR_percent'] - x1) ** 2).sum()

slope = numerator / denominator
intercept = y1 - slope * x1

print("\nTrained Linear Regression Equation:")
print("Views =", round(slope, 2), "* CTR +", round(intercept, 2))


# ----------------------------------------------------
# STEP 7: Make a prediction
# ----------------------------------------------------

new_ctr = 8.0
predicted_views = model.predict([[new_ctr]])

print("\nPredicted views for CTR =", new_ctr, "%:")
print(int(predicted_views[0]))


# ----------------------------------------------------
# STEP 8: Check model accuracy
# ----------------------------------------------------

r2_score = model.score(X_test, y_test)

print("\nModel Accuracy (R² score):", round(r2_score, 2))


# ----------------------------------------------------
# STEP 9: Visualize the results
# ----------------------------------------------------

plt.scatter(df['CTR_percent'], df['Total_Views'], label='Actual Data')
plt.plot(df['CTR_percent'], model.predict(X), label='Regression Line')
plt.scatter(new_ctr, predicted_views, label='Prediction (8% CTR)', s=100)

plt.xlabel("CTR (%)")
plt.ylabel("Total Views")
plt.title("YouTube CTR vs Total Views (Linear Regression)")
plt.legend()
plt.grid(True)
plt.show()
