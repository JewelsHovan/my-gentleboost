# gentleboost/base_estimator.py

import cupy as cp
from typing import Optional
from sklearn.base import BaseEstimator

class DecisionStump(BaseEstimator):
    """A decision stump weak learner for GentleBoost.
    
    This implements a simple decision tree with depth=1 (decision stump)
    optimized for use with GentleBoost algorithm.
    """
    
    def __init__(self, feature_idx: Optional[int] = None, 
             threshold: Optional[float] = None, 
             polarity: Optional[int] = None):
        """Initialize DecisionStump.
        
        Args:
            feature_idx: Index of the feature to split on
            threshold: Threshold value for the split
            polarity: Direction of the split (+1 or -1)
        """
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.polarity = polarity
        
    def get_params(self, deep=True):
        """Get parameters for this estimator.
        
        Args:
            deep: If True, will return the parameters for this estimator
                 and contained subobjects that are estimators.
        
        Returns:
            params: Parameter names mapped to their values.
        """
        return {
            "feature_idx": self.feature_idx,
            "threshold": self.threshold,
            "polarity": self.polarity
        }
    
    def set_params(self, **parameters):
        """Set the parameters of this estimator.
        
        Args:
            **parameters: Estimator parameters.
            
        Returns:
            self: Estimator instance
        """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def fit(self, X: cp.ndarray, y: cp.ndarray, sample_weight: Optional[cp.ndarray] = None) -> 'DecisionStump':
        """Fit the decision stump using weighted samples on GPU.
        
        Args:
            X: Training data of shape (n_samples, n_features) on GPU
            y: Target values {-1, 1} of shape (n_samples,) on GPU
            sample_weight: Sample weights of shape (n_samples,) on GPU
            
        Returns:
            self: Fitted estimator
        """
        if sample_weight is None:
            sample_weight = cp.ones(X.shape[0], dtype=X.dtype)
            
        n_samples, n_features = X.shape
        min_error = float('inf')
        best_feature = 0
        best_threshold = 0
        best_polarity = 1

        # Use fewer percentiles for faster training
        n_thresholds = min(10, X.shape[0] // 100)  # Adaptive threshold count
        percentiles = cp.linspace(0, 100, n_thresholds)
        
        # Vectorize error computation
        for feature in range(n_features):
            sorted_idx = cp.argsort(X[:, feature])
            sorted_x = X[sorted_idx, feature]
            sorted_y = y[sorted_idx]
            sorted_weights = sample_weight[sorted_idx]
            
            thresholds = cp.percentile(sorted_x, percentiles)
            
            # Reshape sorted_y to match broadcasting requirements
            sorted_y = sorted_y.reshape(-1, 1)  # Shape: (n_samples, 1)
            sorted_weights = sorted_weights.reshape(-1, 1)  # Shape: (n_samples, 1)
            
            # Vectorized error computation
            predictions = cp.ones((n_samples, len(thresholds)), dtype=X.dtype)  # Changed shape
            for i, threshold in enumerate(thresholds):
                predictions[:, i] = cp.where(sorted_x <= threshold, -1, 1)
                
            # Compute errors for all thresholds at once
            weighted_errors_pos = cp.sum(sorted_weights * (predictions != sorted_y), axis=0)
            weighted_errors_neg = cp.sum(sorted_weights * (-predictions != sorted_y), axis=0)
            
            # Find best threshold
            min_error_pos = cp.min(weighted_errors_pos)
            min_error_neg = cp.min(weighted_errors_neg)
            
            if min_error_pos < min_error:
                min_error = min_error_pos
                best_feature = feature
                best_threshold = thresholds[cp.argmin(weighted_errors_pos)]
                best_polarity = 1
                
            if min_error_neg < min_error:
                min_error = min_error_neg
                best_feature = feature
                best_threshold = thresholds[cp.argmin(weighted_errors_neg)]
                best_polarity = -1
        
        self.feature_idx = best_feature
        self.threshold = best_threshold
        self.polarity = best_polarity
        
        return self
    
    def predict(self, X: cp.ndarray) -> cp.ndarray:
        """Predict class labels for samples in X using GPU.
        
        Args:
            X: Samples of shape (n_samples, n_features) on GPU
            
        Returns:
            y_pred: Predicted labels {-1, 1} of shape (n_samples,) on GPU
        """
        if self.feature_idx is None:
            raise ValueError("Estimator must be fitted before making predictions")
            
        predictions = cp.ones(X.shape[0], dtype=X.dtype)
        predictions[X[:, self.feature_idx] <= self.threshold] = -1
        return predictions * self.polarity
