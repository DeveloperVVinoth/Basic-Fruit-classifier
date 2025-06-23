import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import joblib

# Create dataset
data = {
    "Weight": [150, 170, 140, 130, 200, 210],
    "Size": [7, 8, 6, 5, 9, 10],
    "Fruit": ["Apple", "Apple", "Apple", "Apple", "Orange", "Orange"]
}

df = pd.DataFrame(data)
X = df[["Weight", "Size"]]
y = df["Fruit"]

# Train model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

# Save model
joblib.dump(model, "model/knn_model.pkl")
print("✅ KNN model trained and saved.")
