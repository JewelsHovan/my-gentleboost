# tests/test_gentleboost.py

import unittest
import cupy as cp
from cuml.datasets import make_classification
import sys
sys.path.append('/home/jhovan/Documents/CustomImplementations/gentleboost')
from gentleboost import GentleBoost

def test_fit_predict(self):
    X, y = make_classification(n_samples=100, n_features=10)
    X = cp.array(X)
    y = cp.array(y)
    # Transform labels from {0,1} to {-1,1}
    y = 2 * y - 1
    
    gb = GentleBoost(n_estimators=10)
    gb.fit(X, y)
    preds = gb.predict(X)
    
    accuracy = cp.mean(preds == y)
    self.assertGreaterEqual(accuracy.get(), 0.5)

if __name__ == '__main__':
    unittest.main()
