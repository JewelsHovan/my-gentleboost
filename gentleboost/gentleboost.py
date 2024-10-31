# gentleboost/gentleboost.py

import cupy as cp
from gentleboost.base_estimator import DecisionStump
from sklearn.base import clone

class GentleBoost:
    def __init__(self, n_estimators=50, learning_rate=1.0, base_estimator=None, patience=10, min_learning_rate=1e-4):
        """Initialize GentleBoost classifier.
        
        Args:
            n_estimators: Maximum number of estimators (default: 50)
            learning_rate: Learning rate for the algorithm (default: 1.0)
            base_estimator: Base weak learner (default: DecisionStump)
            patience: Number of iterations to wait for improvement before early stopping (default: 5)
            min_learning_rate: Minimum learning rate (default: 1e-4)
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.base_estimator = base_estimator or DecisionStump()
        self.patience = patience
        self.learners = []
        self.alphas = []
        self.training_errors_ = []

    def _get_learning_rate(self, iteration):
        """Get the effective learning rate for this iteration"""
        return 0.5 * max(
            self.learning_rate * (0.95 ** iteration),
            self.min_learning_rate
        )

    def fit(self, X, y, validation_fraction=0.2):
        """
        Fit the GentleBoost classifier with validation.
        
        Args:
            X: Training data on GPU (cp.ndarray)
            y: Labels on GPU, should be {-1, 1} (cp.ndarray)
            validation_fraction: Fraction of data to use for validation
        """
        # Ensure y is in {-1, 1}
        if cp.any((y != 1) & (y != -1)):
            raise ValueError("y should contain only -1 and 1")

        n_samples = X.shape[0]
        indices = cp.random.permutation(n_samples)
        n_val = int(n_samples * validation_fraction)
        
        # Split data
        train_idx, val_idx = indices[n_val:], indices[:n_val]
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Initialize separate prediction arrays for training and validation
        F_train = cp.zeros(len(X_train), dtype=X.dtype)
        F_val = cp.zeros(len(X_val), dtype=X.dtype)
        
        # Initialize weights for training set
        weights = cp.ones(len(X_train), dtype=X.dtype) / len(X_train)
        
        # Early stopping setup
        validation_errors = []
        best_iter = 0
        min_val_error = float('inf')
        patience_counter = 0
        improvement_threshold = 1e-4
        
        for iter_num in range(self.n_estimators):
            # Fit weak learner
            learner = clone(self.base_estimator)
            learner.fit(X_train, y_train, sample_weight=weights)
            self.learners.append(learner)
            
            # Get predictions from weak learner
            h_train = learner.predict(X_train)
            h_val = learner.predict(X_val)
            
            # Current learning rate
            alpha = self._get_learning_rate(iter_num)
            self.alphas.append(alpha)
            
            # Update predictions
            F_train += alpha * h_train
            F_val += alpha * h_val
            
            # Update weights with numerical stability
            weights = cp.exp(-y_train * F_train)
            weights = cp.clip(weights, 1e-10, 1e10)
            weights /= cp.sum(weights)
            
            # Calculate training and validation errors
            train_error = float(cp.mean(cp.sign(F_train) != y_train))
            val_error = float(cp.mean(cp.sign(F_val) != y_val))
            
            self.training_errors_.append(train_error)
            validation_errors.append(val_error)
            
            # Early stopping logic using validation error
            if val_error < (min_val_error - improvement_threshold):
                min_val_error = val_error
                best_iter = iter_num
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Print progress
            if iter_num % 10 == 0:
                print(f"Iteration {iter_num}, Train Error: {train_error:.4f}, Val Error: {val_error:.4f}")
            
            if patience_counter >= self.patience:
                print(f"Early stopping at iteration {iter_num}. Best val error: {min_val_error:.4f}")
                # Optionally trim excess learners
                self.learners = self.learners[:best_iter+1]
                self.alphas = self.alphas[:best_iter+1]
                break

    def predict(self, X):
        """
        Predict class labels for samples in X.
        
        Args:
            X: Samples on GPU (cp.ndarray)
        Returns:
            Predicted labels {-1, 1} (cp.ndarray)
        """
        # Pre-allocate array for predictions
        F = cp.zeros(X.shape[0], dtype=X.dtype)
        
        # Batch predictions for memory efficiency
        batch_size = 10000
        for i in range(0, X.shape[0], batch_size):
            batch_end = min(i + batch_size, X.shape[0])
            batch_X = X[i:batch_end]
            
            # Compute predictions for batch
            batch_F = cp.zeros(batch_end - i, dtype=X.dtype)
            for learner, alpha in zip(self.learners, self.alphas):
                batch_F += alpha * learner.predict(batch_X)
                
            F[i:batch_end] = batch_F
        
        return cp.sign(F)
    
    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X.
        
        Args:
            X: Samples on GPU (cp.ndarray)
        Returns:
            Class probabilities (cp.ndarray)
        """
        F = cp.zeros(X.shape[0], dtype=X.dtype)
        
        for learner in self.learners:
            F += learner.predict(X)
            
        # Convert to probabilities using sigmoid function
        proba = 1 / (1 + cp.exp(-2 * F))
        return cp.vstack((1 - proba, proba)).T
