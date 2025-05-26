"""
Simple Word2Vec Data Processing - Educational Implementation
Step 5: Training Pair Generation (Skip-gram and CBOW)
"""

import random
import torch
from typing import List, Tuple, Optional, Dict, Union
from torch.utils.data import Dataset, DataLoader
from collections import Counter

# Import our previous modules
from vocab_builder import SimpleVocabulary
from sub_sampler import SimpleSubSampler
from neg_sampler import SimpleNegativeSampler


class SimpleWord2VecDataset(Dataset):
    """
    Simple PyTorch Dataset for Word2Vec training.
    Generates training pairs for both Skip-gram and CBOW models.
    
    Skip-gram: Predicts context words from center word
    CBOW: Predicts center word from context words
    """
    
    def __init__(self, 
                 sentences: List[List[int]],
                 vocab: SimpleVocabulary,
                 model_type: str = 'skipgram',
                 window_size: int = 5,
                 negative_sampler: Optional[SimpleNegativeSampler] = None):
        """
        Initialize dataset.
        
        Args:
            sentences: List of sentences (each sentence is list of word IDs)
            vocab: SimpleVocabulary instance
            model_type: 'skipgram' or 'cbow'
            window_size: Context window size
            negative_sampler: Optional negative sampler for negative sampling
        """
        self.sentences = sentences
        self.vocab = vocab
        self.model_type = model_type.lower()
        self.window_size = window_size
        self.negative_sampler = negative_sampler
        
        if self.model_type not in ['skipgram', 'cbow']:
            raise ValueError("model_type must be 'skipgram' or 'cbow'")
        
        # Generate all training pairs
        self.training_pairs = self._generate_training_pairs()
        
        print(f"\n{'='*60}")
        print(f"DATASET CREATED")
        print(f"{'='*60}")
        print(f"Model type: {self.model_type.upper()}")
        print(f"Window size: {self.window_size}")
        print(f"Sentences: {len(self.sentences):,}")
        print(f"Training pairs: {len(self.training_pairs):,}")
        print(f"Negative sampling: {'Yes' if self.negative_sampler else 'No'}")
        if self.negative_sampler:
            print(f"Negatives per positive: {self.negative_sampler.num_negatives}")
        print("-" * 60)
    
    def _generate_training_pairs(self) -> List[Tuple]:
        """
        Generate all training pairs from sentences.
        
        Returns:
            List of training pairs (format depends on model type)
        """
        pairs = []
        
        for sentence in self.sentences:
            if len(sentence) < 2:  # Skip very short sentences
                continue
            
            sentence_pairs = self._generate_pairs_from_sentence(sentence)
            pairs.extend(sentence_pairs)
        
        return pairs
    
    def _generate_pairs_from_sentence(self, sentence: List[int]) -> List[Tuple]:
        """
        Generate training pairs from a single sentence.
        
        Args:
            sentence: List of word IDs
            
        Returns:
            List of training pairs from this sentence
        """
        pairs = []
        
        for center_idx in range(len(sentence)):
            center_word = sentence[center_idx]
            
            # Dynamic window size (random between 1 and window_size)
            actual_window = random.randint(1, self.window_size)
            
            # Get context window boundaries
            start = max(0, center_idx - actual_window)
            end = min(len(sentence), center_idx + actual_window + 1)
            
            # Collect context words (excluding center word)
            context_words = []
            for i in range(start, end):
                if i != center_idx:
                    context_words.append(sentence[i])
            
            if not context_words:  # Skip if no context
                continue
            
            # Generate pairs based on model type
            if self.model_type == 'skipgram':
                # Skip-gram: center word → each context word
                for context_word in context_words:
                    pair = self._create_skipgram_pair(center_word, context_word)
                    pairs.append(pair)
            
            elif self.model_type == 'cbow':
                # CBOW: all context words → center word
                pair = self._create_cbow_pair(context_words, center_word)
                pairs.append(pair)
        
        return pairs
    
    def _create_skipgram_pair(self, center_word: int, context_word: int) -> Tuple:
        """
        Create a Skip-gram training pair.
        
        Args:
            center_word: Center word ID
            context_word: Context word ID
            
        Returns:
            Training pair tuple
        """
        if self.negative_sampler:
            # With negative sampling: (center, context, negatives)
            negatives = self.negative_sampler.sample_negatives(context_word)
            return (center_word, context_word, negatives)
        else:
            # Without negative sampling: (center, context)
            return (center_word, context_word)
    
    def _create_cbow_pair(self, context_words: List[int], center_word: int) -> Tuple:
        """
        Create a CBOW training pair.
        
        Args:
            context_words: List of context word IDs
            center_word: Center word ID
            
        Returns:
            Training pair tuple
        """
        if self.negative_sampler:
            # With negative sampling: (context_list, center, negatives)
            negatives = self.negative_sampler.sample_negatives(center_word)
            return (context_words, center_word, negatives)
        else:
            # Without negative sampling: (context_list, center)
            return (context_words, center_word)
    
    def __len__(self) -> int:
        """Return number of training pairs."""
        return len(self.training_pairs)
    
    def __getitem__(self, idx: int) -> Tuple:
        """Get training pair by index."""
        return self.training_pairs[idx]


# =============================================================================
# COLLATE FUNCTIONS FOR BATCHING
# =============================================================================

def simple_collate_skipgram(batch: List[Tuple]) -> Union[Tuple[torch.Tensor, torch.Tensor], 
                                                       Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Simple collate function for Skip-gram batches.
    
    Args:
        batch: List of training pairs
        
    Returns:
        Batched tensors for Skip-gram training
    """
    if len(batch[0]) == 2:  # Without negative sampling
        center_words, context_words = zip(*batch)
        center_tensor = torch.tensor(center_words, dtype=torch.long)
        context_tensor = torch.tensor(context_words, dtype=torch.long)
        return center_tensor, context_tensor
    
    elif len(batch[0]) == 3:  # With negative sampling
        center_words, context_words, negatives = zip(*batch)
        
        center_tensor = torch.tensor(center_words, dtype=torch.long)
        context_tensor = torch.tensor(context_words, dtype=torch.long)
        
        # Flatten negatives and create corresponding center words
        neg_flat = []
        neg_centers = []
        for i, neg_list in enumerate(negatives):
            neg_flat.extend(neg_list)
            neg_centers.extend([center_words[i]] * len(neg_list))
        
        neg_tensor = torch.tensor(neg_flat, dtype=torch.long)
        neg_center_tensor = torch.tensor(neg_centers, dtype=torch.long)
        
        return center_tensor, context_tensor, neg_tensor, neg_center_tensor


def simple_collate_cbow(batch: List[Tuple]) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                                                   Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Simple collate function for CBOW batches.
    
    Args:
        batch: List of training pairs
        
    Returns:
        Batched tensors for CBOW training
    """
    if len(batch[0]) == 2:  # Without negative sampling
        context_lists, center_words = zip(*batch)
        
        # Pad context lists to same length
        max_context_len = max(len(ctx) for ctx in context_lists)
        context_padded = []
        context_masks = []
        
        for ctx in context_lists:
            padded = ctx + [0] * (max_context_len - len(ctx))
            mask = [1] * len(ctx) + [0] * (max_context_len - len(ctx))
            context_padded.append(padded)
            context_masks.append(mask)
        
        context_tensor = torch.tensor(context_padded, dtype=torch.long)
        context_mask_tensor = torch.tensor(context_masks, dtype=torch.float)
        center_tensor = torch.tensor(center_words, dtype=torch.long)
        
        return context_tensor, context_mask_tensor, center_tensor
    
    elif len(batch[0]) == 3:  # With negative sampling
        context_lists, center_words, negatives = zip(*batch)
        
        # Pad context lists
        max_context_len = max(len(ctx) for ctx in context_lists)
        context_padded = []
        context_masks = []
        
        for ctx in context_lists:
            padded = ctx + [0] * (max_context_len - len(ctx))
            mask = [1] * len(ctx) + [0] * (max_context_len - len(ctx))
            context_padded.append(padded)
            context_masks.append(mask)
        
        context_tensor = torch.tensor(context_padded, dtype=torch.long)
        context_mask_tensor = torch.tensor(context_masks, dtype=torch.float)
        center_tensor = torch.tensor(center_words, dtype=torch.long)
        
        # Handle negatives
        neg_flat = []
        neg_contexts = []
        neg_masks = []
        
        for i, neg_list in enumerate(negatives):
            for _ in neg_list:
                neg_flat.extend(neg_list)
                neg_contexts.append(context_padded[i])
                neg_masks.append(context_masks[i])
        
        neg_tensor = torch.tensor(neg_flat, dtype=torch.long)
        neg_context_tensor = torch.tensor(neg_contexts, dtype=torch.long)
        neg_mask_tensor = torch.tensor(neg_masks, dtype=torch.float)
        
        return context_tensor, context_mask_tensor, center_tensor, neg_tensor, neg_context_tensor, neg_mask_tensor


# =============================================================================
# UTILITY FUNCTIONS FOR DATASET ANALYSIS
# =============================================================================

def convert_sentences_to_ids(tokenized_sentences: List[List[str]], vocab: SimpleVocabulary) -> List[List[int]]:
    """
    Convert tokenized sentences to word ID sentences.
    
    Args:
        tokenized_sentences: List of tokenized sentences
        vocab: SimpleVocabulary instance
        
    Returns:
        List of sentences with word IDs
    """
    id_sentences = []
    unk_count = 0
    total_words = 0
    
    for sentence in tokenized_sentences:
        id_sentence = []
        for word in sentence:
            word_id = vocab.get_word_id(word)
            id_sentence.append(word_id)
            
            total_words += 1
            if vocab.get_word(word_id) == '<UNK>':
                unk_count += 1
        
        if id_sentence:  # Only add non-empty sentences
            id_sentences.append(id_sentence)
    
    print(f"\nSentence conversion complete:")
    print(f"  - Total words: {total_words:,}")
    print(f"  - UNK words: {unk_count:,} ({unk_count/total_words*100:.1f}%)")
    print(f"  - Sentences: {len(id_sentences):,}")
    
    return id_sentences


def print_training_pair_examples(dataset: SimpleWord2VecDataset, num_examples: int = 10) -> None:
    """
    Print examples of training pairs in human-readable format.
    
    Args:
        dataset: SimpleWord2VecDataset instance
        num_examples: Number of examples to show
    """
    print(f"\n{'='*60}")
    print(f"TRAINING PAIR EXAMPLES")
    print(f"{'='*60}")
    print(f"Model: {dataset.model_type.upper()}")
    print(f"Window size: {dataset.window_size}")
    print(f"Showing {min(num_examples, len(dataset))} examples out of {len(dataset):,} total")
    print("-" * 60)
    
    for i in range(min(num_examples, len(dataset))):
        pair = dataset[i]
        
        if dataset.model_type == 'skipgram':
            if len(pair) == 2:  # Without negative sampling
                center_id, context_id = pair
                center_word = dataset.vocab.get_word(center_id)
                context_word = dataset.vocab.get_word(context_id)
                print(f"Example {i+1}: '{center_word}' → '{context_word}'")
            
            elif len(pair) == 3:  # With negative sampling
                center_id, context_id, negatives = pair
                center_word = dataset.vocab.get_word(center_id)
                context_word = dataset.vocab.get_word(context_id)
                negative_words = [dataset.vocab.get_word(neg_id) for neg_id in negatives]
                print(f"Example {i+1}: '{center_word}' → '{context_word}' | Negatives: {negative_words}")
        
        elif dataset.model_type == 'cbow':
            if len(pair) == 2:  # Without negative sampling
                context_ids, center_id = pair
                context_words = [dataset.vocab.get_word(ctx_id) for ctx_id in context_ids]
                center_word = dataset.vocab.get_word(center_id)
                print(f"Example {i+1}: {context_words} → '{center_word}'")
            
            elif len(pair) == 3:  # With negative sampling
                context_ids, center_id, negatives = pair
                context_words = [dataset.vocab.get_word(ctx_id) for ctx_id in context_ids]
                center_word = dataset.vocab.get_word(center_id)
                negative_words = [dataset.vocab.get_word(neg_id) for neg_id in negatives]
                print(f"Example {i+1}: {context_words} → '{center_word}' | Negatives: {negative_words}")
    
    print("-" * 60)


def print_sentence_to_pairs_demo(sentence: List[str], 
                                vocab: SimpleVocabulary,
                                model_type: str = 'skipgram',
                                window_size: int = 2,
                                negative_sampler: Optional[SimpleNegativeSampler] = None) -> None:
    """
    Demonstrate how a single sentence gets converted to training pairs.
    
    Args:
        sentence: List of words (tokens)
        vocab: SimpleVocabulary instance
        model_type: 'skipgram' or 'cbow'
        window_size: Context window size
        negative_sampler: Optional negative sampler
    """
    print(f"\n{'='*60}")
    print(f"SENTENCE → TRAINING PAIRS DEMONSTRATION")
    print(f"{'='*60}")
    print(f"Original sentence: {' '.join(sentence)}")
    print(f"Model type: {model_type.upper()}")
    print(f"Window size: {window_size}")
    print("-" * 60)
    
    # Convert to word IDs
    sentence_ids = [vocab.get_word_id(word) for word in sentence]
    
    print("Word → ID mapping:")
    for word, word_id in zip(sentence, sentence_ids):
        actual_word = vocab.get_word(word_id)
        if word != actual_word:
            print(f"  '{word}' → {word_id} ('{actual_word}')")
        else:
            print(f"  '{word}' → {word_id}")
    print()
    
    # Create mini dataset with just this sentence
    mini_dataset = SimpleWord2VecDataset(
        sentences=[sentence_ids],
        vocab=vocab,
        model_type=model_type,
        window_size=window_size,
        negative_sampler=negative_sampler
    )
    
    print(f"Generated {len(mini_dataset)} training pairs:")
    print("-" * 40)
    
    # Show all pairs from this sentence
    for i, pair in enumerate(mini_dataset.training_pairs):
        if model_type == 'skipgram':
            if len(pair) == 2:
                center_id, context_id = pair
                center_word = vocab.get_word(center_id)
                context_word = vocab.get_word(context_id)
                print(f"Pair {i+1}: '{center_word}' → '{context_word}'")
            elif len(pair) == 3:
                center_id, context_id, negatives = pair
                center_word = vocab.get_word(center_id)
                context_word = vocab.get_word(context_id)
                negative_words = [vocab.get_word(neg_id) for neg_id in negatives]
                print(f"Pair {i+1}: '{center_word}' → '{context_word}' | Neg: {negative_words}")
        
        elif model_type == 'cbow':
            if len(pair) == 2:
                context_ids, center_id = pair
                context_words = [vocab.get_word(ctx_id) for ctx_id in context_ids]
                center_word = vocab.get_word(center_id)
                print(f"Pair {i+1}: {context_words} → '{center_word}'")
            elif len(pair) == 3:
                context_ids, center_id, negatives = pair
                context_words = [vocab.get_word(ctx_id) for ctx_id in context_ids]
                center_word = vocab.get_word(center_id)
                negative_words = [vocab.get_word(neg_id) for neg_id in negatives]
                print(f"Pair {i+1}: {context_words} → '{center_word}' | Neg: {negative_words}")
    
    print("-" * 60)


def analyze_dataset_statistics(dataset: SimpleWord2VecDataset) -> None:
    """
    Analyze and display comprehensive dataset statistics.
    
    Args:
        dataset: SimpleWord2VecDataset instance
    """
    print(f"\n{'='*60}")
    print(f"DATASET STATISTICS")
    print(f"{'='*60}")
    
    # Basic info
    print(f"Dataset Configuration:")
    print(f"  - Model type: {dataset.model_type.upper()}")
    print(f"  - Window size: {dataset.window_size}")
    print(f"  - Vocabulary size: {dataset.vocab.vocab_size:,}")
    print(f"  - Input sentences: {len(dataset.sentences):,}")
    print(f"  - Training pairs: {len(dataset.training_pairs):,}")
    print(f"  - Negative sampling: {'Yes' if dataset.negative_sampler else 'No'}")
    
    # Sentence length analysis
    sentence_lengths = [len(sentence) for sentence in dataset.sentences]
    if sentence_lengths:
        print(f"\nSentence Length Statistics:")
        print(f"  - Average length: {sum(sentence_lengths) / len(sentence_lengths):.1f} words")
        print(f"  - Min length: {min(sentence_lengths)} words")
        print(f"  - Max length: {max(sentence_lengths)} words")
        print(f"  - Total words: {sum(sentence_lengths):,}")
    
    # Pairs per sentence analysis
    pairs_per_sentence = []
    for sentence in dataset.sentences:
        mini_dataset = SimpleWord2VecDataset(
            sentences=[sentence],
            vocab=dataset.vocab,
            model_type=dataset.model_type,
            window_size=dataset.window_size,
            negative_sampler=None  # For counting, ignore negatives
        )
        pairs_per_sentence.append(len(mini_dataset.training_pairs))
    
    if pairs_per_sentence:
        print(f"\nTraining Pairs per Sentence:")
        print(f"  - Average pairs: {sum(pairs_per_sentence) / len(pairs_per_sentence):.1f}")
        print(f"  - Min pairs: {min(pairs_per_sentence)}")
        print(f"  - Max pairs: {max(pairs_per_sentence)}")
    
    # Word frequency in training pairs
    if dataset.model_type == 'skipgram':
        center_words = []
        context_words = []
        for pair in dataset.training_pairs:
            center_words.append(pair[0])
            context_words.append(pair[1])
        
        center_counter = Counter(center_words)
        context_counter = Counter(context_words)
        
        print(f"\nSkip-gram Pair Analysis:")
        print(f"  - Unique center words: {len(center_counter)}")
        print(f"  - Unique context words: {len(context_counter)}")
        print(f"  - Most frequent center word: '{dataset.vocab.get_word(center_counter.most_common(1)[0][0])}' ({center_counter.most_common(1)[0][1]} times)")
        print(f"  - Most frequent context word: '{dataset.vocab.get_word(context_counter.most_common(1)[0][0])}' ({context_counter.most_common(1)[0][1]} times)")
    
    elif dataset.model_type == 'cbow':
        center_words = []
        all_context_words = []
        context_sizes = []
        
        for pair in dataset.training_pairs:
            context_list = pair[0]
            center_word = pair[1]
            
            center_words.append(center_word)
            all_context_words.extend(context_list)
            context_sizes.append(len(context_list))
        
        center_counter = Counter(center_words)
        context_counter = Counter(all_context_words)
        
        print(f"\nCBOW Pair Analysis:")
        print(f"  - Unique center words: {len(center_counter)}")
        print(f"  - Average context size: {sum(context_sizes) / len(context_sizes):.1f}")
        print(f"  - Min context size: {min(context_sizes)}")
        print(f"  - Max context size: {max(context_sizes)}")
        print(f"  - Most frequent center word: '{dataset.vocab.get_word(center_counter.most_common(1)[0][0])}' ({center_counter.most_common(1)[0][1]} times)")
    
    print("-" * 60)


def demonstrate_batching(dataset: SimpleWord2VecDataset, batch_size: int = 4) -> None:
    """
    Demonstrate how training pairs get batched for training.
    
    Args:
        dataset: SimpleWord2VecDataset instance
        batch_size: Batch size for demonstration
    """
    print(f"\n{'='*60}")
    print(f"BATCHING DEMONSTRATION")
    print(f"{'='*60}")
    print(f"Model: {dataset.model_type.upper()}")
    print(f"Batch size: {batch_size}")
    print("-" * 60)
    
    # Create DataLoader
    if dataset.model_type == 'skipgram':
        collate_fn = simple_collate_skipgram
    else:
        collate_fn = simple_collate_cbow
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Show first batch
    first_batch = next(iter(dataloader))
    
    print(f"Batch tensors:")
    for i, tensor in enumerate(first_batch):
        print(f"  Tensor {i+1}: shape {tensor.shape}, dtype {tensor.dtype}")
        print(f"           values: {tensor}")
    
    print(f"\nHuman-readable batch contents:")
    if dataset.model_type == 'skipgram':
        if len(first_batch) == 2:  # Without negatives
            center_tensor, context_tensor = first_batch
            print(f"{'Center':<15} → {'Context'}")
            print("-" * 30)
            for center_id, context_id in zip(center_tensor, context_tensor):
                center_word = dataset.vocab.get_word(center_id.item())
                context_word = dataset.vocab.get_word(context_id.item())
                print(f"{center_word:<15} → {context_word}")
        
        elif len(first_batch) == 4:  # With negatives
            center_tensor, context_tensor, neg_tensor, neg_center_tensor = first_batch
            print(f"Positive pairs:")
            print(f"{'Center':<15} → {'Context'}")
            print("-" * 30)
            for center_id, context_id in zip(center_tensor, context_tensor):
                center_word = dataset.vocab.get_word(center_id.item())
                context_word = dataset.vocab.get_word(context_id.item())
                print(f"{center_word:<15} → {context_word}")
            
            print(f"\nNegative pairs:")
            print(f"{'Center':<15} → {'Negative'}")
            print("-" * 30)
            for center_id, neg_id in zip(neg_center_tensor, neg_tensor):
                center_word = dataset.vocab.get_word(center_id.item())
                neg_word = dataset.vocab.get_word(neg_id.item())
                print(f"{center_word:<15} → {neg_word}")
    
    elif dataset.model_type == 'cbow':
        if len(first_batch) == 3:  # Without negatives
            context_tensor, context_mask, center_tensor = first_batch
            print(f"{'Context':<25} → {'Center'}")
            print("-" * 35)
            for ctx_ids, mask, center_id in zip(context_tensor, context_mask, center_tensor):
                # Get actual context words (excluding padding)
                context_words = []
                for ctx_id, m in zip(ctx_ids, mask):
                    if m > 0:  # Not padding
                        context_words.append(dataset.vocab.get_word(ctx_id.item()))
                center_word = dataset.vocab.get_word(center_id.item())
                print(f"{str(context_words):<25} → {center_word}")
    
    print("-" * 60)


# =============================================================================
# EXAMPLE USAGE AND TESTING
# =============================================================================

if __name__ == "__main__":
    # Import required modules
    try:
        from vocab_builder import SimpleVocabulary
        from neg_sampler import SimpleNegativeSampler
    except ImportError:
        print("This module requires SimpleVocabulary and SimpleNegativeSampler.")
        exit(1)
    
    # Sample data for testing
    sample_sentences = [
        ['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog'],
        ['the', 'dog', 'is', 'very', 'lazy', 'and', 'sleeps', 'all', 'day'],
        ['word2vec', 'learns', 'word', 'embeddings', 'from', 'text', 'data'],
        ['the', 'algorithm', 'uses', 'the', 'skip', 'gram', 'model'],
        ['training', 'pairs', 'are', 'generated', 'from', 'sliding', 'windows'],
        ['context', 'words', 'help', 'predict', 'the', 'center', 'word'],
        ['skip', 'gram', 'predicts', 'context', 'from', 'center', 'word'],
        ['cbow', 'predicts', 'center', 'from', 'context', 'words']
    ]
    
    print("Testing SimpleWord2VecDataset with sample data...")
    
    # Build vocabulary
    vocab = SimpleVocabulary(min_freq=1, max_size=None)
    vocab.build_from_sentences(sample_sentences)
    
    # Convert to IDs
    id_sentences = convert_sentences_to_ids(sample_sentences, vocab)
    
    # Test both model types
    for model_type in ['skipgram', 'cbow']:
        print(f"\n{'#'*80}")
        print(f"TESTING {model_type.upper()} MODEL")
        print(f"{'#'*80}")
        
        # Test without negative sampling
        print(f"\n--- WITHOUT NEGATIVE SAMPLING ---")
        dataset = SimpleWord2VecDataset(
            sentences=id_sentences,
            vocab=vocab,
            model_type=model_type,
            window_size=2,
            negative_sampler=None
        )
        
        print_training_pair_examples(dataset, num_examples=8)
        analyze_dataset_statistics(dataset)
        demonstrate_batching(dataset, batch_size=4)
        
        # Test with negative sampling
        print(f"\n--- WITH NEGATIVE SAMPLING ---")
        neg_sampler = SimpleNegativeSampler(vocab, num_negatives=3)
        dataset_with_neg = SimpleWord2VecDataset(
            sentences=id_sentences,
            vocab=vocab,
            model_type=model_type,
            window_size=2,
            negative_sampler=neg_sampler
        )
        
        print_training_pair_examples(dataset_with_neg, num_examples=5)
        demonstrate_batching(dataset_with_neg, batch_size=3)
        
        # Demonstrate single sentence conversion
        demo_sentence = ['the', 'quick', 'brown', 'fox', 'jumps']
        print_sentence_to_pairs_demo(
            sentence=demo_sentence,
            vocab=vocab,
            model_type=model_type,
            window_size=2,
            negative_sampler=neg_sampler
        )
    
    print(f"\n{'='*60}")
    print("SimpleWord2VecDataset testing complete!")
    print(f"{'='*60}")