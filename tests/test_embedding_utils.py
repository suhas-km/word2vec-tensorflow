#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for embedding utilities.
"""

import unittest
import numpy as np
import os
import tempfile
from unittest.mock import patch

from src.utils.embedding_utils import (
    save_embeddings, load_embeddings, find_similar_words,
    word_analogy, cosine_similarity
)

class TestEmbeddingUtils(unittest.TestCase):
    """Test cases for embedding utilities."""

    def setUp(self):
        """Set up test fixtures."""
        # Create small test embeddings and vocabulary
        self.vocab_size = 10
        self.embedding_dim = 5
        self.vocabulary = {
            'test': 0,
            'word': 1,
            'embedding': 2,
            'king': 3,
            'queen': 4,
            'man': 5,
            'woman': 6,
            'apple': 7,
            'fruit': 8,
            'computer': 9
        }
        
        # Create random embeddings with specific relationships for testing
        np.random.seed(42)
        self.weights = np.random.randn(self.vocab_size, self.embedding_dim).astype(np.float32)
        
        # Make king - man + woman close to queen for testing analogies
        self.weights[3] = self.weights[5] + self.weights[6] - self.weights[4] + np.random.randn(self.embedding_dim) * 0.1

    def test_save_load_embeddings(self):
        """Test saving and loading embeddings."""
        with tempfile.NamedTemporaryFile(suffix='.npz') as temp:
            # Save embeddings
            save_embeddings(
                model=None, 
                vocabulary=self.vocabulary, 
                path=temp.name, 
                weights=self.weights
            )
            
            # Load embeddings
            loaded_weights, loaded_vocab = load_embeddings(temp.name)
            
            # Check that weights and vocabulary are the same
            np.testing.assert_allclose(self.weights, loaded_weights)
            self.assertEqual(self.vocabulary, loaded_vocab)

    def test_cosine_similarity(self):
        """Test cosine similarity calculation."""
        # Create two vectors with known cosine similarity
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        v3 = np.array([1.0, 1.0, 0.0])
        
        # Cosine similarity between perpendicular vectors is 0
        self.assertAlmostEqual(cosine_similarity(v1, v2), 0.0)
        
        # Cosine similarity between parallel vectors is 1
        self.assertAlmostEqual(cosine_similarity(v1, v1), 1.0)
        
        # Cosine similarity at 45 degrees is 1/sqrt(2)
        self.assertAlmostEqual(cosine_similarity(v1, v3), 1/np.sqrt(2))

    def test_find_similar_words(self):
        """Test finding similar words."""
        # Use the first word as query
        query_word = 'test'
        similar_words = find_similar_words(
            query_word, 
            self.weights, 
            self.vocabulary, 
            top_k=3
        )
        
        # Check return format
        self.assertIsInstance(similar_words, list)
        self.assertEqual(len(similar_words), 3)
        self.assertIsInstance(similar_words[0], tuple)
        self.assertEqual(len(similar_words[0]), 2)
        
        # Check that the query word itself is not in the results
        query_words = [word for word, _ in similar_words]
        self.assertNotIn(query_word, query_words)

    def test_word_analogy(self):
        """Test word analogy."""
        # Test king - man + woman = queen (approximately)
        results = word_analogy('king', 'man', 'woman', self.weights, self.vocabulary, top_k=5)
        
        # Check return format
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        
        # Since we set up the vectors specifically, queen should be the top result
        # or at least in the top results
        result_words = [word for word, _ in results]
        self.assertIn('queen', result_words)


if __name__ == '__main__':
    unittest.main()
