# examples/demo.py

import cupy as cp
from cuml.datasets import make_classification
from gentleboost import GentleBoost
from cuml.linear_model import LogisticRegression

# Generate synthetic data
X, y = make_classification(n_samples=1000, n_features=20)
X = cp.array(X)
y = cp.array(y)

# Initialize GentleBoost with a custom base estimator
base_estimator = LogisticRegression()
gb = GentleBoost(n_estimators=100, base_estimator=base_estimator)

# Train the model
gb.fit(X, y)

# Make predictions
preds = gb.predict(X)

# Evaluate accuracy
accuracy = cp.mean(preds == y)
print(f'Accuracy: {accuracy.get():.2f}')
