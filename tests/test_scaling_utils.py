#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for scaling utilities.
"""

import unittest
import numpy as np
import tensorflow as tf
import tempfile
import os
import shutil

from src.utils.scaling_utils import (
    create_text_vectorizer,
    get_vocabulary_and_inverse,
    generate_training_data,
    create_efficient_dataset
)

class TestScalingUtils(unittest.TestCase):
    """Test cases for scaling utilities."""

    def setUp(self):
        """Set up test fixtures."""
        # Create sample sentences
        self.sentences = [
            "This is the first test sentence",
            "Another test sentence with different words",
            "A third sentence with some overlapping words",
            "Test sentence with numbers 123 and punctuation!",
            "Final test sentence to complete the dataset"
        ]
        self.vocab_size = 30
        
        # Set random seed for reproducibility
        tf.random.set_seed(42)
        np.random.seed(42)

    def test_create_text_vectorizer(self):
        """Test creation of text vectorizer."""
        # Create text vectorizer
        vectorize_layer = create_text_vectorizer(self.sentences, self.vocab_size)
        
        # Check properties
        self.assertIsNotNone(vectorize_layer)
        self.assertEqual(vectorize_layer.max_tokens, self.vocab_size)
        
        # Check vocabulary
        vocab = vectorize_layer.get_vocabulary()
        self.assertLessEqual(len(vocab), self.vocab_size)
        
        # Check that common words are in vocabulary
        self.assertIn("test", vocab)
        self.assertIn("sentence", vocab)

    def test_get_vocabulary_and_inverse(self):
        """Test getting vocabulary and inverse vocabulary mapping."""
        # Create text vectorizer
        vectorize_layer = create_text_vectorizer(self.sentences, self.vocab_size)
        
        # Get vocabulary and inverse
        vocabulary, inverse_vocab = get_vocabulary_and_inverse(vectorize_layer)
        
        # Check types and structure
        self.assertIsInstance(vocabulary, dict)
        self.assertIsInstance(inverse_vocab, dict)
        
        # Check that they are inverse of each other
        for word, idx in vocabulary.items():
            self.assertEqual(word, inverse_vocab[idx])

    def test_generate_training_data(self):
        """Test generation of training data from sequences."""
        # Create sequences
        vectorize_layer = create_text_vectorizer(self.sentences, self.vocab_size)
        sequences = [vectorize_layer(s) for s in self.sentences]
        
        # Generate training data
        targets, contexts, labels = generate_training_data(
            sequences=sequences,
            vocab_size=self.vocab_size,
            window_size=2,
            num_ns=4
        )
        
        # Check shapes
        self.assertGreater(len(targets), 0)
        self.assertEqual(len(targets), len(contexts))
        self.assertEqual(len(targets), len(labels))
        
        # Check types
        self.assertEqual(targets.dtype, tf.int64)
        self.assertEqual(contexts.dtype, tf.int64)
        self.assertEqual(labels.dtype, tf.int64)

    def test_create_efficient_dataset(self):
        """Test creation of efficient dataset."""
        # Create sequences
        vectorize_layer = create_text_vectorizer(self.sentences, self.vocab_size)
        sequences = [vectorize_layer(s) for s in self.sentences]
        
        # Create dataset
        dataset = create_efficient_dataset(
            sequences=sequences,
            vocab_size=self.vocab_size,
            window_size=2,
            num_ns=4,
            batch_size=2,
            buffer_size=100
        )
        
        # Check that it's a dataset
        self.assertIsInstance(dataset, tf.data.Dataset)
        
        # Check a batch from the dataset
        for targets, (contexts, labels) in dataset.take(1):
            # Check types
            self.assertEqual(targets.dtype, tf.int64)
            self.assertEqual(contexts.dtype, tf.int64)
            self.assertEqual(labels.dtype, tf.int64)
            
            # Check shapes - batch_size x ? 
            self.assertEqual(targets.shape[0], 2)  # batch size of 2
            self.assertEqual(contexts.shape[0], 2)
            self.assertEqual(labels.shape[0], 2)


if __name__ == '__main__':
    unittest.main()
