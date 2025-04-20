import kagglehub
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Download latest version
path = kagglehub.dataset_download("fedesoriano/body-fat-prediction-dataset")

print("Path to dataset files:", path)

# Load the dataset CSV file from the downloaded path
csv_file = os.path.join(path, "bodyfat.csv")
df = pd.read_csv(csv_file)

# Create engineered features
df["BMI"] = df["Weight"] / (df["Height"] ** 2) * 703  # standard BMI in lbs/inches
df["WaistHipRatio"] = df["Abdomen"] / df["Hip"]

selected_features = ["Wrist", "BMI", "WaistHipRatio", "Neck", "Forearm", "Thigh", "Hip", "Biceps", "Ankle"]

# Drop 'BodyFat' (target variable) and 'Density' (used to derive BodyFat via Siri's equation)
# Including 'Density' would cause data leakage, as BodyFat is calculated directly from it:
# BodyFat = (495 / Density) - 450
X_sel = df[selected_features]
y = df["BodyFat"]

# Split data into training and testing sets
# test_size=0.2 means 20% of the data is used for testing, which is a common default that balances
# training performance with generalization ability
# random_state=42 is set for reproducibility – it ensures the same data split every time the code runs
X_train, X_test, y_train, y_test = train_test_split(X_sel, y, test_size=0.2, random_state=42)

# Initialize and train a simple linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate model performance
print("R² Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))