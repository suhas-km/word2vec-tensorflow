from setuptools import setup, find_packages

setup(
    name="word2vec-tensorflow",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "tensorflow>=2.0.0",
        "numpy",
        "matplotlib",
        "scikit-learn",
        "tqdm",
    ],
    author="Suhas KM",
    author_email="suhaskm@example.com",
    description="Word2Vec implementation in TensorFlow with utilities for training, evaluation, and scaling",
    keywords="word2vec, tensorflow, nlp, embeddings",
    url="https://github.com/suhaskm-neu/word2vec-tensorflow",
    python_requires=">=3.6",
)
