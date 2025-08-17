# Day 5: Lasso Regression – Feature Selection for Salary Prediction

# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 2. Create Synthetic Dataset
np.random.seed(42)
data_size = 120

experience = np.random.randint(1, 15, data_size)
education_level = np.random.randint(1, 4, data_size)  # 1=Bachelor, 2=Master, 3=PhD
age = np.random.randint(22, 60, data_size)
certifications = np.random.randint(0, 5, data_size)
hours_per_week = np.random.randint(30, 50, data_size)
random_noise = np.random.normal(0, 5000, data_size)

# True salary formula (some features less impactful)
salary = (experience * 3000) + (education_level * 5000) + \
         (age * 100) + (certifications * 2000) + random_noise

df = pd.DataFrame({
    "Experience": experience,
    "EducationLevel": education_level,
    "Age": age,
    "Certifications": certifications,
    "HoursPerWeek": hours_per_week,  # Less important feature
    "Salary": salary
})

# 3. Split Data
X = df.drop("Salary", axis=1)
y = df["Salary"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Train Lasso Regression Model
model = Lasso(alpha=1000)  # Higher alpha = more regularization
model.fit(X_train, y_train)

# 5. Predict
y_pred = model.predict(X_test)

# 6. Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# 7. Visualization: Feature Importance
plt.bar(X.columns, model.coef_, color='orange')
plt.xlabel("Features")
plt.ylabel("Coefficient Value")
plt.title("Feature Importance via Lasso Regression")
plt.show()
