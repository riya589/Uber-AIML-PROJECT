# ==============================================
# ANALYSIS AND PREDICTION OF UBER TRIP PATTERNS
# ==============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder

# ---------------------------
# 1. Load and Prepare Dataset
# ---------------------------
df = pd.read_csv("Uber-Jan-Feb-FOIL (1).csv")

# Convert date to datetime
df['date'] = pd.to_datetime(df['date'])

# Feature Engineering
df['Day'] = df['date'].dt.day
df['DayOfWeek'] = df['date'].dt.dayofweek
df['Month'] = df['date'].dt.month
df['Hour'] = df['date'].dt.hour  # Keep for visuals (will be 0 in daily data but placeholder for structure)

# Encode categorical base
le = LabelEncoder()
df['Base'] = le.fit_transform(df['dispatching_base_number'])

# ---------------------------
# 2. Model Training
# ---------------------------
X = df[['Base', 'active_vehicles', 'Day', 'DayOfWeek', 'Month']]
y = df['trips']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ---------------------------
# 3. Model Evaluation
# ---------------------------
r2 = metrics.r2_score(y_test, y_pred)
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("MODEL PERFORMANCE METRICS")
print(f"R-squared (R²): {r2:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# ---------------------------
# 4. Data Visualizations
# ---------------------------

# Total Pickups by Hour of the Day (placeholder: all 0s if dataset lacks hours)
plt.figure(figsize=(10,5))
sns.barplot(x='Hour', y='trips', data=df, estimator=sum, ci=None, palette="viridis")
plt.title('Total Pickups by Hour of the Day', fontsize=14)
plt.xlabel('Hour of the Day')
plt.ylabel('Total Trips')
plt.grid(alpha=0.3)
plt.show()

# Total Pickups by Day of the Week
plt.figure(figsize=(10,5))
sns.barplot(x='DayOfWeek', y='trips', data=df, estimator=sum, ci=None, palette="plasma")
plt.title('Total Pickups by Day of the Week', fontsize=14)
plt.xlabel('Day of the Week (0 = Monday, 6 = Sunday)')
plt.ylabel('Total Trips')
plt.grid(alpha=0.3)
plt.show()

# Heatmap of Pickups by Hour and Day of Week
# For daily dataset, this will appear mostly empty in Hour axis (since all Hour=0),
# but structure is included for consistency with report.
pivot = df.pivot_table(values='trips', index='DayOfWeek', columns='Hour', aggfunc='sum', fill_value=0)
plt.figure(figsize=(10,6))
sns.heatmap(pivot, cmap="YlGnBu")
plt.title('Heatmap of Pickups by Hour and Day of Week', fontsize=14)
plt.xlabel('Hour of the Day')
plt.ylabel('Day of the Week (0 = Monday)')
plt.show()

# ---------------------------
# 5. Feature Importance Plot
# ---------------------------
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(8,5))
sns.barplot(x='Importance', y='Feature', data=feature_importance, palette="mako")
plt.title('Feature Importance in Prediction Model', fontsize=14)
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.grid(alpha=0.3)
plt.show()
