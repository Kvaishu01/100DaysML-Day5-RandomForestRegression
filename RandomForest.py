import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Generate synthetic dataset
np.random.seed(42)
data_size = 200
data = {
    "PM2.5": np.random.uniform(10, 150, data_size),
    "PM10": np.random.uniform(20, 200, data_size),
    "NO2": np.random.uniform(5, 80, data_size),
    "SO2": np.random.uniform(2, 40, data_size),
    "CO": np.random.uniform(0.1, 3, data_size),
    "O3": np.random.uniform(10, 120, data_size)
}
df = pd.DataFrame(data)
df["AQI"] = (0.4*df["PM2.5"] + 0.3*df["PM10"] + 
             0.1*df["NO2"] + 0.05*df["SO2"] +
             0.05*df["CO"]*100 + 0.1*df["O3"]) + np.random.normal(0, 10, data_size)

# Features and target
X = df.drop("AQI", axis=1)
y = df["AQI"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Visualization
plt.figure(figsize=(7,5))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual AQI")
plt.ylabel("Predicted AQI")
plt.title("Random Forest Regression - AQI Prediction")
plt.show()
