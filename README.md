# Word2Vec TensorFlow Implementation

A comprehensive implementation of Word2Vec using TensorFlow, featuring both skip-gram with negative sampling and CBOW approaches. This project extends the [official TensorFlow Word2Vec tutorial](https://www.tensorflow.org/text/tutorials/word2vec) with additional utilities for training, evaluation, and scaling to larger datasets.

## 📚 Repository Structure

```
word2vec-tensorflow/
├── data/               # Data storage directory
│   └── shakespeare/    # Shakespeare corpus data
├── notebooks/         # Jupyter notebooks
│   ├── word2vec_tensorflow.ipynb  # Skip-gram implementation
│   └── word2vec_tf_2_0.ipynb      # CBOW implementation
├── src/               # Source code
│   ├── models/        # Model implementations
│   │   └── word2vec.py            # Word2Vec model class
│   ├── utils/         # Utility functions
│   │   ├── embedding_utils.py     # Embedding evaluation utilities
│   │   └── scaling_utils.py       # Scaling utilities for large datasets
│   └── train.py       # Main training script
├── checkpoints/       # Model checkpoints (git-ignored)
├── outputs/           # Output files (embeddings, visualizations)
├── tests/             # Test scripts
├── docs/              # Documentation
├── .gitignore        # Git ignore file
├── setup.py          # Package installation file
└── README.md         # This file
```

## ✨ Features

- **Multiple Word2Vec Architectures**:
  - Skip-gram with negative sampling (main implementation)
  - Continuous Bag of Words (CBOW) implementation

- **Training Capabilities**:
  - Efficient data pipeline with tf.data API
  - TensorBoard integration for monitoring training
  - Checkpointing and early stopping

- **Embedding Evaluation**:
  - Cosine similarity for finding similar words
  - Word analogy tasks (e.g., king - man + woman = queen)
  - t-SNE visualization of embeddings

- **Scalability**:
  - Process multiple text files from directories
  - Memory-efficient data generation
  - Optimized batching and prefetching

## 🚀 Getting Started

### Installation

There are two ways to install this project:

1. **Install as a package**:

```bash
# Clone the repository
git clone https://github.com/suhaskm-neu/word2vec-tensorflow.git
cd word2vec-tensorflow

# Install in development mode
pip install -e .
```

2. **Install dependencies only**:

```bash
pip install tensorflow numpy matplotlib scikit-learn tqdm
```

### Training Models

To train a model using the provided training script:

#### Training on Shakespeare Text (Default)

```bash
python -m src.train --mode shakespeare --embedding-dim 128 --epochs 5
```

#### Training on a Custom Corpus

```bash
python -m src.train --mode large --corpus-dir /path/to/text/files --vocab-size 10000 --embedding-dim 300
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|--------|
| `--mode` | Training mode (`shakespeare` or `large`) | `shakespeare` |
| `--corpus-dir` | Directory containing text corpus for large mode | None |
| `--embedding-dim` | Embedding dimension | 128 |
| `--vocab-size` | Vocabulary size | 4096 |
| `--epochs` | Number of training epochs | 5 |
| `--output-dir` | Directory to save outputs | `outputs/` |

## 📊 Evaluating Embeddings

After training, the script automatically performs evaluation:

1. **Word Similarities**: Finds semantically similar words based on cosine similarity
2. **Word Analogies**: Tests relationships like "king is to man as queen is to woman"
3. **Visualizations**: Generates t-SNE plots of the embedding space

## 📝 Example Output

```
Words most similar to 'king':
  queen: 0.8532
  prince: 0.8349
  throne: 0.7851
  royal: 0.7723
  monarch: 0.7546

Word analogies:
king is to man as queen is to: woman (0.8721)
```

## 🔍 Advanced Usage

The implementation includes several utilities that can be imported in your own projects:

### Using Embedding Utilities

```python
# After installing the package
from src.utils.embedding_utils import find_similar_words, visualize_embeddings, word_analogy

# Load previously saved embeddings
weights, vocab = load_embeddings('outputs/shakespeare_embeddings.npz')

# Find similar words
similar_words = find_similar_words('king', weights, vocab, top_k=10)
print(similar_words)

# Perform word analogies (king - man + woman = queen)
analogies = word_analogy('king', 'man', 'woman', weights, vocab)
print(analogies)

# Visualize word embeddings
visualize_embeddings(
    weights, 
    vocab, 
    words_to_plot=['king', 'queen', 'man', 'woman', 'prince', 'princess']
)
```

### Training Custom Models

```python
from src.utils.scaling_utils import (
    load_large_corpus_from_directory, create_text_vectorizer,
    get_vocabulary_and_inverse, create_efficient_dataset, train_with_checkpointing
)
from src.models.word2vec import load_or_create_model

# Load your custom corpus
sentences = load_large_corpus_from_directory('path/to/corpus_dir')

# Create vocabulary
vocab_size = 10000
vectorize_layer = create_text_vectorizer(sentences, vocab_size)
vocabulary, inverse_vocab = get_vocabulary_and_inverse(vectorize_layer)

# Vectorize sentences
sequences = [vectorize_layer(s) for s in sentences]

# Create dataset with skip-gram pairs and negative samples
dataset = create_efficient_dataset(
    sequences, 
    vocab_size=vocab_size, 
    window_size=4, 
    num_ns=5,  # Number of negative samples
    batch_size=1024
)

# Create model
model = load_or_create_model('my_model.weights.h5', vocab_size, embedding_dim=300)

# Train with checkpointing
train_with_checkpointing(
    model, 
    dataset, 
    epochs=10, 
    checkpoint_dir='./my_checkpoints'
)
```

## 🔗 References

- [Original TensorFlow Word2Vec Tutorial](https://www.tensorflow.org/text/tutorials/word2vec)
- [Original Word2Vec Paper (Mikolov et al.)](https://arxiv.org/abs/1301.3781)
- [TensorFlow Documentation](https://www.tensorflow.org/)

## 📓 Original Notebooks

- [Main Notebook](https://github.com/suhaskm-neu/word2vec-tensorflow/blob/main/word2vec_tensorflow.ipynb)
