# Word2Vec TensorFlow Implementation

A comprehensive implementation of Word2Vec using TensorFlow, featuring both skip-gram with negative sampling and CBOW approaches. This project extends the [official TensorFlow Word2Vec tutorial](https://www.tensorflow.org/text/tutorials/word2vec) with additional utilities for training, evaluation, and scaling to larger datasets.

## 📚 Repository Contents

- **Notebooks**:
  - `word2vec_tensorflow.ipynb`: Main implementation of skip-gram with negative sampling
  - `word2vec_tf_2_0.ipynb`: Simple CBOW implementation using TensorFlow 2.x

- **Python Utilities**:
  - `embedding_utils.py`: Functions for evaluating and visualizing trained embeddings
  - `scaling_utils.py`: Tools for scaling to larger datasets and efficient training
  - `word2vec_demo.py`: Demo script showing how to train and evaluate embeddings

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

### Prerequisites

```bash
pip install tensorflow numpy matplotlib scikit-learn tqdm
```

### Training on Shakespeare Text

```bash
python word2vec_demo.py --mode shakespeare --embedding-dim 128 --epochs 5
```

### Training on a Custom Corpus

```bash
python word2vec_demo.py --mode large --corpus-dir /path/to/text/files --vocab-size 10000 --embedding-dim 300
```

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

```python
from embedding_utils import find_similar_words, visualize_embeddings, word_analogy
from scaling_utils import load_large_corpus_from_directory, train_with_checkpointing
```

## 🔗 References

- [Original TensorFlow Word2Vec Tutorial](https://www.tensorflow.org/text/tutorials/word2vec)
- [Original Word2Vec Paper (Mikolov et al.)](https://arxiv.org/abs/1301.3781)
- [TensorFlow Documentation](https://www.tensorflow.org/)

## 📓 Original Notebooks

- [Main Notebook](https://github.com/suhaskm-neu/word2vec-tensorflow/blob/main/word2vec_tensorflow.ipynb)
