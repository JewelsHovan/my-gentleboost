# gentleboost/base_estimator.py

import cupy as cp
from typing import Optional
from sklearn.base import BaseEstimator

class DecisionStump(BaseEstimator):
    """A decision stump weak learner for GentleBoost.
    
    This implements a simple decision tree with depth=1 (decision stump)
    optimized for use with GentleBoost algorithm.
    """
    
    def __init__(self):
        """Initialize regression stump for GentleBoost."""
        self.feature_idx = None
        self.threshold = None
        self.left_value = None
        self.right_value = None
    
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
            "left_value": self.left_value,
            "right_value": self.right_value
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
        """Fit regression stump using weighted least squares."""
        if sample_weight is None:
            sample_weight = cp.ones(X.shape[0], dtype=X.dtype)
            
        n_samples, n_features = X.shape
        min_error = float('inf')
        
        # Use fewer percentiles for faster training
        n_thresholds = min(10, X.shape[0] // 100)
        percentiles = cp.linspace(0, 100, n_thresholds)
        
        for feature in range(n_features):
            x_feature = X[:, feature]
            thresholds = cp.percentile(x_feature, percentiles)
            
            # Vectorize over thresholds
            x_feature_expanded = x_feature[:, cp.newaxis]
            thresholds_expanded = thresholds[cp.newaxis, :]
            left_mask = x_feature_expanded <= thresholds_expanded
            right_mask = ~left_mask
            
            # Exclude invalid thresholds
            valid_thresholds = (cp.sum(left_mask, axis=0) > 0) & (cp.sum(right_mask, axis=0) > 0)
            if not cp.any(valid_thresholds):
                continue
            left_mask = left_mask[:, valid_thresholds]
            right_mask = right_mask[:, valid_thresholds]
            thresholds_valid = thresholds[valid_thresholds]
            
            # Vectorized weighted calculations
            w_left = sample_weight[:, cp.newaxis] * left_mask
            w_right = sample_weight[:, cp.newaxis] * right_mask
            sum_w_left = cp.sum(w_left, axis=0)
            sum_w_right = cp.sum(w_right, axis=0)
            
            y_left = y[:, cp.newaxis] * left_mask
            y_right = y[:, cp.newaxis] * right_mask
            
            sum_wy_left = cp.sum(w_left * y_left, axis=0)
            sum_wy_right = cp.sum(w_right * y_right, axis=0)
            
            left_value = sum_wy_left / sum_w_left
            right_value = sum_wy_right / sum_w_right
            
            # Compute errors vectorized
            error_left = cp.sum(w_left * (y_left - left_value[cp.newaxis, :]) ** 2, axis=0)
            error_right = cp.sum(w_right * (y_right - right_value[cp.newaxis, :]) ** 2, axis=0)
            total_error = error_left + error_right
            
            # Update best parameters
            min_error_idx = cp.argmin(total_error)
            if total_error[min_error_idx] < min_error:
                min_error = total_error[min_error_idx]
                self.feature_idx = feature
                self.threshold = thresholds_valid[min_error_idx]
                self.left_value = left_value[min_error_idx]
                self.right_value = right_value[min_error_idx]
        
        return self
    
    def predict(self, X: cp.ndarray) -> cp.ndarray:
        """Predict continuous values."""
        if self.feature_idx is None:
            raise ValueError("Estimator must be fitted before making predictions")
            
        predictions = cp.where(
            X[:, self.feature_idx] <= self.threshold,
            self.left_value,
            self.right_value
        )
        return predictions
