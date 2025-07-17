#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run all unit tests for the Word2Vec project.
"""

import unittest
import sys
import os

if __name__ == "__main__":
    # Add the parent directory to the path so we can import from src
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    # Discover and run all tests
    loader = unittest.TestLoader()
    start_dir = './tests'
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Exit with error code if tests failed
    sys.exit(not result.wasSuccessful())
