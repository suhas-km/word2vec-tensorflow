#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Basic tests for Word2Vec model functionality.
"""

import unittest
import numpy as np
import tensorflow as tf

from src.models.word2vec import Word2Vec, create_model

class TestWord2VecModel(unittest.TestCase):
    """Test cases for Word2Vec model."""

    def setUp(self):
        """Set up test fixtures."""
        self.vocab_size = 100
        self.embedding_dim = 8
        self.model = Word2Vec(self.vocab_size, self.embedding_dim)
        
        # Create random target and context words
        self.batch_size = 4
        self.targets = tf.constant([1, 10, 20, 30], dtype=tf.int64)
        self.contexts = tf.constant([5, 15, 25, 35], dtype=tf.int64)

    def test_model_initialization(self):
        """Test model initialization and architecture."""
        # Check that model layers were created correctly
        self.assertIsNotNone(self.model.target_embedding)
        self.assertIsNotNone(self.model.context_embedding)
        
        # Check embedding dimensions
        self.assertEqual(self.model.target_embedding.input_dim, self.vocab_size)
        self.assertEqual(self.model.target_embedding.output_dim, self.embedding_dim)
        self.assertEqual(self.model.context_embedding.input_dim, self.vocab_size)
        self.assertEqual(self.model.context_embedding.output_dim, self.embedding_dim)

    def test_call_method(self):
        """Test forward pass of the model."""
        # Call the model
        outputs = self.model((self.targets, self.contexts))
        
        # Check output shape
        self.assertEqual(outputs.shape, (self.batch_size, 1))

    def test_create_model_function(self):
        """Test the create_model utility function."""
        model = create_model(self.vocab_size, self.embedding_dim)
        
        # Check model was created and compiled
        self.assertIsNotNone(model.optimizer)
        self.assertIsNotNone(model.loss)
        
        # Test forward pass
        outputs = model((self.targets, self.contexts))
        self.assertEqual(outputs.shape, (self.batch_size, 1))


if __name__ == '__main__':
    unittest.main()
