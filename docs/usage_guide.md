# Word2Vec TensorFlow Usage Guide

This guide provides detailed instructions on using the Word2Vec implementation in this repository.

## Table of Contents

1. [Installation](#installation)
2. [Basic Usage](#basic-usage)
3. [Training on Shakespeare Corpus](#training-on-shakespeare-corpus)
4. [Training on Custom Corpus](#training-on-custom-corpus)
5. [Advanced Usage](#advanced-usage)
6. [Evaluation Methods](#evaluation-methods)
7. [Tips for Better Embeddings](#tips-for-better-embeddings)
8. [API Reference](#api-reference)

## Installation

### Option 1: Install as a package

```bash
# Clone the repository
git clone https://github.com/suhaskm-neu/word2vec-tensorflow.git
cd word2vec-tensorflow

# Install in development mode
pip install -e .
```

### Option 2: Install dependencies only

```bash
pip install tensorflow numpy matplotlib scikit-learn tqdm
```

## Basic Usage

The simplest way to get started is using the provided training script:

```bash
# Train on Shakespeare corpus
python -m src.train --mode shakespeare

# Train on your own corpus
python -m src.train --mode large --corpus-dir /path/to/text/files
```

## Training on Shakespeare Corpus

The Shakespeare corpus is a small dataset suitable for testing and rapid experimentation:

```bash
python -m src.train --mode shakespeare --embedding-dim 128 --epochs 5
```

This will:
1. Download the Shakespeare corpus
2. Preprocess the text into sentences
3. Build a vocabulary (default size: 4096)
4. Create training pairs with negative sampling
5. Train the Word2Vec model
6. Save embeddings to `outputs/shakespeare_embeddings.npz`
7. Print similar words and analogies
8. Visualize the embeddings using t-SNE

## Training on Custom Corpus

For larger datasets, use the large corpus mode:

```bash
python -m src.train --mode large \
    --corpus-dir /path/to/text/files \
    --vocab-size 10000 \
    --embedding-dim 300 \
    --epochs 10
```

The custom corpus should be a directory containing text files. Each file will be processed and all sentences will be extracted for training.

## Advanced Usage

### Using Word2Vec in Your Own Scripts

```python
from src.models.word2vec import Word2Vec, create_model

# Create a new model
vocab_size = 10000
embedding_dim = 300
model = Word2Vec(vocab_size, embedding_dim)

# Or use the helper function to create and compile the model
model = create_model(vocab_size, embedding_dim)

# Train the model
model.fit(dataset, epochs=10)

# Access the embeddings
embeddings = model.target_embedding.get_weights()[0]
```

### Loading and Using Trained Embeddings

```python
from src.utils.embedding_utils import load_embeddings, find_similar_words

# Load embeddings from a saved file
weights, vocab = load_embeddings('outputs/shakespeare_embeddings.npz')

# Find similar words
similar_words = find_similar_words('king', weights, vocab, top_k=10)
for word, similarity in similar_words:
    print(f"{word}: {similarity:.4f}")
```

### Working with Large Corpora

```python
from src.utils.scaling_utils import load_large_corpus_from_directory

# Load text from multiple files
sentences = load_large_corpus_from_directory('/path/to/corpus')

# Preprocess and train as usual
```

## Evaluation Methods

### Word Similarity

Finding semantically similar words using cosine similarity:

```python
from src.utils.embedding_utils import find_similar_words

similar_words = find_similar_words('king', weights, vocab, top_k=10)
```

### Word Analogies

Testing analogical reasoning (king - man + woman = queen):

```python
from src.utils.embedding_utils import word_analogy

results = word_analogy('king', 'man', 'woman', weights, vocab)
```

### Visualization

Visualizing word relationships using t-SNE:

```python
from src.utils.embedding_utils import visualize_embeddings

visualize_embeddings(
    weights, 
    vocab,
    words_to_plot=['king', 'queen', 'man', 'woman', 'prince', 'princess']
)
```

## Tips for Better Embeddings

1. **Corpus Size**: Larger corpora generally produce better embeddings
2. **Preprocessing**: Clean and normalize text properly (lowercase, remove rare words)
3. **Hyperparameters**:
   - Embedding dimension: 100-300 is typical (larger for bigger corpora)
   - Window size: 5-10 works well for most applications
   - Negative samples: 5-20 is common (more for larger datasets)
4. **Training Duration**: More epochs help, but with diminishing returns
5. **Evaluation**: Always evaluate using multiple metrics

## API Reference

### Models (`src.models.word2vec`)

- `Word2Vec(vocab_size, embedding_dim)`: Main model class
- `create_model(vocab_size, embedding_dim)`: Creates and compiles a model
- `load_or_create_model(model_path, vocab_size, embedding_dim)`: Loads existing or creates new model

### Embedding Utilities (`src.utils.embedding_utils`)

- `save_embeddings(model, vocabulary, path, weights=None)`: Save embeddings to file
- `load_embeddings(path)`: Load embeddings from file
- `find_similar_words(word, weights, vocabulary, top_k=10)`: Find similar words
- `word_analogy(word1, word2, word3, weights, vocabulary, top_k=10)`: Perform word analogy
- `visualize_embeddings(weights, vocabulary, words_to_plot=None)`: Visualize embeddings

### Scaling Utilities (`src.utils.scaling_utils`)

- `load_large_corpus_from_directory(directory)`: Load text from multiple files
- `create_text_vectorizer(sentences, vocab_size)`: Create a text vectorization layer
- `get_vocabulary_and_inverse(vectorize_layer)`: Get vocabulary and inverse mapping
- `generate_training_data(sequences, vocab_size, window_size, num_ns)`: Generate training data
- `create_efficient_dataset(sequences, vocab_size, window_size, num_ns, batch_size, buffer_size=None)`: Create efficient TensorFlow dataset
- `train_with_checkpointing(model, dataset, epochs, checkpoint_dir)`: Train with checkpointing
