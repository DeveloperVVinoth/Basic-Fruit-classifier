import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import joblib

# Sample training data
data = {
    "Weight": [150, 170, 140, 130, 200, 210],
    "Size": [7, 8, 6, 5, 9, 10],
    "Fruit": ["Apple", "Apple", "Apple", "Apple", "Orange", "Orange"]
}

df = pd.DataFrame(data)

# Features (X) and target (y)
X = df[["Weight", "Size"]]
y = df["Fruit"]

# Train KNN model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

# Save the model
joblib.dump(model, "model/fruit_knn_model.pkl")
print("✅ Fruit KNN model saved successfully.")
