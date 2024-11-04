# tests/test_gentleboost.py

import unittest
import cupy as cp
import numpy as np
from cuml.datasets import make_classification
import sys
sys.path.append('/home/jhovan/Documents/CustomImplementations/gentleboost')
from gentleboost import GentleBoost
from gentleboost.base_estimator import DecisionStump

class TestDecisionStump(unittest.TestCase):
    def setUp(self):
        # Create simple dataset for testing
        self.X = cp.array([
            [1.0, 2.0],
            [2.0, 1.0],
            [3.0, 4.0],
            [4.0, 3.0]
        ])
        self.y = cp.array([1, -1, 1, -1])
        self.weights = cp.array([0.25, 0.25, 0.25, 0.25])
        
    def test_decision_stump_initialization(self):
        stump = DecisionStump()
        self.assertIsNone(stump.feature_idx)
        self.assertIsNone(stump.threshold)
        self.assertIsNone(stump.left_value)
        self.assertIsNone(stump.right_value)
    
    def test_decision_stump_fit_predict(self):
        # Create a more clearly separable dataset
        self.X = cp.array([
            [1.0, 5.0],
            [2.0, 4.0],
            [3.0, 6.0],
            [4.0, 3.0],
            [5.0, 2.0],
            [6.0, 1.0]
        ])
        self.y = cp.array([1, 1, 1, -1, -1, -1])
        self.weights = cp.ones(6) / 6

        stump = DecisionStump()
        stump.fit(self.X, self.y, self.weights)
        
        # Debug prints
        print("\nDebug information:")
        print(f"Feature idx: {stump.feature_idx}")
        print(f"Threshold: {stump.threshold}")
        print(f"Left value: {stump.left_value}")
        print(f"Right value: {stump.right_value}")
        
        # Check if parameters were set
        self.assertIsNotNone(stump.feature_idx)
        self.assertIsNotNone(stump.threshold)
        self.assertIsNotNone(stump.left_value)
        self.assertIsNotNone(stump.right_value)
        
        # Test predictions
        preds = stump.predict(self.X)
        self.assertEqual(preds.shape, self.y.shape)
        
        # Additional test to verify prediction quality
        accuracy = cp.mean(cp.sign(preds) == self.y)
        print(f"Prediction accuracy: {accuracy}")
        self.assertGreater(accuracy.get(), 0.5)
    
    def test_decision_stump_params(self):
        stump = DecisionStump()
        stump.fit(self.X, self.y)
        
        # Test get_params
        params = stump.get_params()
        self.assertIn('feature_idx', params)
        self.assertIn('threshold', params)
        self.assertIn('left_value', params)
        self.assertIn('right_value', params)
        
        # Test set_params
        new_params = {
            'feature_idx': 0,
            'threshold': 1.0,
            'left_value': -1.0,
            'right_value': 1.0
        }
        stump.set_params(**new_params)
        self.assertEqual(stump.feature_idx, 0)
        self.assertEqual(stump.threshold, 1.0)

class TestGentleBoost(unittest.TestCase):
    def setUp(self):
        # Create dataset for testing
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        self.X = cp.array(X)
        self.y = cp.array(y)
        # Transform to {-1, 1}
        self.y = 2 * self.y - 1
        
    def test_gentleboost_initialization(self):
        gb = GentleBoost(n_estimators=10)
        self.assertEqual(gb.n_estimators, 10)
        self.assertIsInstance(gb.base_estimator, DecisionStump)
        self.assertEqual(len(gb.learners), 0)
        self.assertEqual(len(gb.training_errors_), 0)
    
    def test_gentleboost_fit_predict(self):
        gb = GentleBoost(n_estimators=10)
        gb.fit(self.X, self.y)
        
        # Check if learners were created
        self.assertEqual(len(gb.learners), 10)
        self.assertEqual(len(gb.training_errors_), 10)
        
        # Test predictions
        preds = gb.predict(self.X)
        self.assertEqual(preds.shape, self.y.shape)
        accuracy = cp.mean(preds == self.y)
        self.assertGreaterEqual(accuracy.get(), 0.5)
    
    def test_invalid_labels(self):
        # Test with invalid labels (not {-1, 1})
        invalid_y = cp.array([0, 1, 2, 3])
        gb = GentleBoost(n_estimators=10)
        with self.assertRaises(ValueError):
            gb.fit(self.X, invalid_y)
    
    def test_prediction_batching(self):
        # Test if batching works correctly
        gb = GentleBoost(n_estimators=5)
        gb.fit(self.X, self.y)
        
        # Test with different batch sizes
        preds1 = gb.predict(self.X)  # default batch size
        
        # Modify batch_size temporarily for testing
        original_batch_size = 10000
        try:
            gb.batch_size = 10  # small batch size
            preds2 = gb.predict(self.X)
            cp.testing.assert_array_equal(preds1, preds2)
        finally:
            gb.batch_size = original_batch_size

if __name__ == '__main__':
    unittest.main()
