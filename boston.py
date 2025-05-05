# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras import layers, models

# 1. Load the dataset
df = pd.read_csv("/content/drive/MyDrive/lp5 dl/boston_housing.csv")

# 2. Data Exploration
print("Data Summary:")
print(df.describe())

print("\nMissing values:")
print(df.isnull().sum())

# 3. Data Visualization
sns.set(style="whitegrid")
features_to_plot = ['rm', 'lstat', 'ptratio', 'nox', 'indus']
plt.figure(figsize=(18, 10))
for idx, feature in enumerate(features_to_plot):
    plt.subplot(2, 3, idx + 1)
    sns.scatterplot(data=df, x=feature, y='MEDV')
    plt.title(f'{feature} vs MEDV')
plt.tight_layout()
plt.show()

# 4. Preprocessing
X = df.drop(columns=["MEDV"])
y = df["MEDV"]

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 5. Build DNN model for regression
model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)  # Output layer for regression
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# 6. Train the model
history = model.fit(X_train, y_train, validation_split=0.2, epochs=100, verbose=1)

# 7. Evaluate the model
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"\nTest Mean Absolute Error (MAE): {test_mae:.2f}")

# 8. Plot training history
plt.figure(figsize=(10, 5))
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Val MAE')
plt.title('MAE during training')
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error')
plt.legend()
plt.show()

# 9. Predict and compare
y_pred = model.predict(X_test).flatten()

plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual MEDV")
plt.ylabel("Predicted MEDV")
plt.title("Actual vs Predicted House Prices")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.show()
