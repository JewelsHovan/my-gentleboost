# gentleboost/gentleboost.py

import cupy as cp
from gentleboost.base_estimator import DecisionStump
from sklearn.base import clone

class GentleBoost:
    def __init__(self, n_estimators=50, base_estimator=None):
        """Initialize GentleBoost classifier.
        
        Args:
            n_estimators: Number of weak learners (default: 50)
            base_estimator: Base weak learner (default: DecisionStump)
        """
        self.n_estimators = n_estimators
        self.base_estimator = base_estimator or DecisionStump()
        self.learners = []
        self.training_errors_ = []

    def fit(self, X, y):
        """Fit the GentleBoost classifier.
        
        Args:
            X: Training data on GPU (cp.ndarray)
            y: Labels on GPU, should be {-1, 1} (cp.ndarray)
        """
        # Ensure y is in {-1, 1}
        if cp.any((y != 1) & (y != -1)):
            raise ValueError("y should contain only -1 and 1")

        n_samples = X.shape[0]
        F = cp.zeros(n_samples, dtype=X.dtype)
        weights = cp.ones(n_samples, dtype=X.dtype) / n_samples
        
        for iter_num in range(self.n_estimators):
            # Create a new instance instead of using clone
            learner = DecisionStump()
            learner.fit(X, y, sample_weight=weights)
            self.learners.append(learner)
            
            # Get predictions from weak learner
            h = learner.predict(X)
            
            # Update F(x)
            F += h
            
            # Update weights (multiplicative update)
            weights *= cp.exp(-y * h)
            weights = cp.clip(weights, 1e-10, 1e10)  # For numerical stability
            weights /= cp.sum(weights)
            
            # Calculate training error
            train_error = float(cp.mean(cp.sign(F) != y))
            self.training_errors_.append(train_error)
            
            # Print progress
            if iter_num % 10 == 0:
                print(f"Iteration {iter_num}, Training Error: {train_error:.4f}")

    def predict(self, X):
        """Predict class labels for samples in X."""
        # Pre-collect all learner parameters
        feature_indices = cp.array([learner.feature_idx for learner in self.learners], dtype=cp.int32)
        thresholds = cp.array([learner.threshold for learner in self.learners], dtype=X.dtype)
        left_values = cp.array([learner.left_value for learner in self.learners], dtype=X.dtype)
        right_values = cp.array([learner.right_value for learner in self.learners], dtype=X.dtype)
        
        F = cp.zeros(X.shape[0], dtype=X.dtype)
        batch_size = 10000
        
        for i in range(0, X.shape[0], batch_size):
            batch_end = min(i + batch_size, X.shape[0])
            batch_X = X[i:batch_end]
            
            # Vectorized predictions for all learners
            batch_X_features = batch_X[:, feature_indices]
            mask = batch_X_features <= thresholds[cp.newaxis, :]
            predictions = cp.where(mask, left_values[cp.newaxis, :], right_values[cp.newaxis, :])
            batch_F = cp.sum(predictions, axis=1)
            F[i:batch_end] = batch_F
        
        return cp.sign(F)
