# Word2Vec Training and Evaluation Guide

This guide shows you how to train a Word2Vec model on your training file and evaluate it on your evaluation dataset.

## 🚀 Quick Start

### 1. Basic Usage (Command Line)

```bash
python train_and_evaluate_word2vec.py \
    --train_file path/to/your/training_data.txt \
    --eval_file path/to/your/evaluation_data.txt \
    --output_dir results \
    --model_type skipgram \
    --embedding_dim 300 \
    --epochs 10 \
    --batch_size 512
```

### 2. Using Configuration File

```bash
python train_and_evaluate_word2vec.py \
    --train_file path/to/training_data.txt \
    --eval_file path/to/evaluation_data.txt \
    --config_file word2vec_config.json \
    --output_dir results
```

### 3. Programmatic Usage

```python
from train_and_evaluate_word2vec import Word2VecExperiment, create_default_config

# Create configuration
config = create_default_config()
config.update({
    'model_type': 'skipgram',
    'embedding_dim': 300,
    'epochs': 10,
    'batch_size': 512,
    'learning_rate': 0.025
})

# Create experiment
experiment = Word2VecExperiment(config)

# Train model
model, history = experiment.train_model('path/to/training_data.txt')

# Evaluate model
eval_results = experiment.evaluate_model('path/to/evaluation_data.txt')

# Save results
experiment.save_results('results')
```

## 📁 Input File Format

Your text files should contain one sentence per line:

**training_data.txt:**
```
The quick brown fox jumps over the lazy dog.
Word embeddings capture semantic relationships between words.
Machine learning models process natural language effectively.
```

**evaluation_data.txt:**
```
Natural language processing enables automated text analysis.
Deep learning architectures learn complex pattern representations.
Semantic similarity measures compare word meaning relationships.
```

## ⚙️ Configuration Options

### Model Parameters
- `model_type`: "skipgram" or "cbow"
- `embedding_dim`: Dimension of word vectors (100, 200, 300, etc.)
- `window_size`: Context window size (3-10)
- `min_word_freq`: Minimum word frequency threshold (1-10)
- `max_vocab_size`: Maximum vocabulary size (null for unlimited)

### Training Parameters
- `epochs`: Number of training epochs (5-20)
- `batch_size`: Training batch size (128, 256, 512, 1024)
- `learning_rate`: Initial learning rate (0.01-0.05)
- `min_learning_rate`: Minimum learning rate (0.0001)

### Advanced Options
- `use_negative_sampling`: Enable negative sampling (recommended: true)
- `num_negatives`: Number of negative samples (3-10)
- `use_subsampling`: Enable frequent word subsampling (recommended: true)
- `subsample_threshold`: Subsampling threshold (1e-3 to 1e-5)

## 📊 Output Files

After training and evaluation, you'll get:

```
results/
├── word2vec_model.pt           # Trained PyTorch model
├── vocabulary.json             # Vocabulary mappings
├── word_embeddings.txt         # Embeddings in text format
├── evaluation_results.json     # Evaluation metrics
└── training_history.json       # Training loss curves
```

### Using Saved Embeddings

```python
# Load embeddings for use in other applications
embeddings = {}
with open('results/word_embeddings.txt', 'r') as f:
    next(f)  # Skip header
    for line in f:
        parts = line.strip().split()
        word = parts[0]
        vector = [float(x) for x in parts[1:]]
        embeddings[word] = vector

# Use embeddings
word_vector = embeddings['example']
```

### Loading Trained Model

```python
import torch
from word2vec_model import Word2VecModel

# Load model checkpoint
checkpoint = torch.load('results/word2vec_model.pt')

# Recreate model
model = Word2VecModel(
    vocab_size=checkpoint['vocab_size'],
    embedding_dim=checkpoint['embedding_dim'],
    model_type=checkpoint['model_type']
)
model.load_state_dict(checkpoint['model_state