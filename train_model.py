# Import libraries
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Create simple dataset
data = {
    "feature": [1, 2, 3, 4, 5],
    "target": [2, 4, 6, 8, 10]
}

df = pd.DataFrame(data)

# Split features and target
X = df[["feature"]]
y = df["target"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
joblib.dump(model, "model.pkl")

print("Model trained and saved as model.pkl")