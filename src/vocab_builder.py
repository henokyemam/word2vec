"""
Simple Word2Vec Data Processing - Educational Implementation
Step 2: Vocabulary Building and Management
"""

from collections import Counter
from typing import List, Dict, Optional


class SimpleVocabulary:
    """
    Simple vocabulary management for Word2Vec.
    Builds word-to-index mappings and tracks word frequencies.
    """
    
    def __init__(self, min_freq: int = 5, max_size: Optional[int] = None):
        """
        Initialize vocabulary builder.
        
        Args:
            min_freq: Minimum frequency for a word to be included
            max_size: Maximum vocabulary size (None for unlimited)
        """
        self.min_freq = min_freq
        self.max_size = max_size
        
        # Core mappings
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self.word_freq: Dict[str, int] = {}
        
        # Raw statistics (before filtering)
        self.word_counts = Counter()
        self.total_words = 0
        
        # Vocabulary info
        self.vocab_size = 0
        self.special_tokens = ['<UNK>']  # Unknown word token
        
        # Build status
        self.is_built = False
    
    def build_from_sentences(self, tokenized_sentences: List[List[str]]) -> None:
        """
        Build vocabulary from tokenized sentences.
        
        Args:
            tokenized_sentences: List of tokenized sentences
        """
        print(f"\n{'='*60}")
        print(f"BUILDING VOCABULARY")
        print(f"{'='*60}")
        print(f"Settings:")
        print(f"  - Minimum frequency: {self.min_freq}")
        print(f"  - Maximum size: {self.max_size or 'unlimited'}")
        print("-" * 60)
        
        # Step 1: Count all words
        print("Step 1: Counting word frequencies...")
        self._count_words(tokenized_sentences)
        
        # Step 2: Filter words by frequency
        print("Step 2: Filtering words by frequency...")
        filtered_words = self._filter_by_frequency()
        
        # Step 3: Apply size limit
        print("Step 3: Applying vocabulary size limit...")
        final_words = self._apply_size_limit(filtered_words)
        
        # Step 4: Build mappings
        print("Step 4: Building word-to-index mappings...")
        self._build_mappings(final_words)
        
        self.is_built = True
        print(f"✓ Vocabulary building complete!")
        print("-" * 60)
    
    def _count_words(self, tokenized_sentences: List[List[str]]) -> None:
        """Count frequency of all words in the corpus."""
        for sentence in tokenized_sentences:
            for word in sentence:
                self.word_counts[word] += 1
                self.total_words += 1
        
        print(f"  - Total words in corpus: {self.total_words:,}")
        print(f"  - Unique words found: {len(self.word_counts):,}")
    
    def _filter_by_frequency(self) -> List[str]:
        """Filter words that appear at least min_freq times."""
        filtered_words = [
            word for word, count in self.word_counts.items() 
            if count >= self.min_freq
        ]
        
        words_removed = len(self.word_counts) - len(filtered_words)
        print(f"  - Words meeting frequency threshold: {len(filtered_words):,}")
        print(f"  - Words filtered out: {words_removed:,}")
        
        return filtered_words
    
    def _apply_size_limit(self, filtered_words: List[str]) -> List[str]:
        """Apply maximum vocabulary size limit."""
        # Sort by frequency (descending)
        sorted_words = sorted(filtered_words, key=lambda w: self.word_counts[w], reverse=True)
        
        if self.max_size:
            # Reserve space for special tokens
            max_regular_words = self.max_size - len(self.special_tokens)
            final_words = sorted_words[:max_regular_words]
            
            if len(sorted_words) > max_regular_words:
                words_dropped = len(sorted_words) - max_regular_words
                print(f"  - Words after size limit: {len(final_words):,}")
                print(f"  - Additional words dropped: {words_dropped:,}")
            else:
                print(f"  - All {len(final_words):,} words fit within size limit")
        else:
            final_words = sorted_words
            print(f"  - No size limit applied: {len(final_words):,} words")
        
        return final_words
    
    def _build_mappings(self, words: List[str]) -> None:
        """Build word-to-index and index-to-word mappings."""
        # Add special tokens first
        for i, token in enumerate(self.special_tokens):
            self.word2idx[token] = i
            self.idx2word[i] = token
            self.word_freq[token] = 0  # Special tokens have 0 frequency
        
        # Add regular words
        start_idx = len(self.special_tokens)
        for i, word in enumerate(words, start_idx):
            self.word2idx[word] = i
            self.idx2word[i] = word
            self.word_freq[word] = self.word_counts[word]
        
        self.vocab_size = len(self.word2idx)
        print(f"  - Final vocabulary size: {self.vocab_size:,}")
        print(f"  - Special tokens: {self.special_tokens}")
    
    def get_word_id(self, word: str) -> int:
        """
        Get word index. Returns <UNK> index if word not in vocabulary.
        
        Args:
            word: Word to look up
            
        Returns:
            Word index
        """
        return self.word2idx.get(word, self.word2idx['<UNK>'])
    
    def get_word(self, idx: int) -> str:
        """
        Get word from index.
        
        Args:
            idx: Word index
            
        Returns:
            Word string
        """
        return self.idx2word.get(idx, '<UNK>')


# =============================================================================
# UTILITY FUNCTIONS FOR VOCABULARY ANALYSIS
# =============================================================================

def print_vocabulary_stats(vocab: SimpleVocabulary) -> None:
    """
    Print comprehensive vocabulary statistics.
    
    Args:
        vocab: SimpleVocabulary instance
    """
    if not vocab.is_built:
        print("Vocabulary not built yet!")
        return
    
    print(f"\n{'='*60}")
    print(f"VOCABULARY STATISTICS")
    print(f"{'='*60}")
    
    # Basic stats
    print(f"Corpus Statistics:")
    print(f"  - Total words in corpus: {vocab.total_words:,}")
    print(f"  - Unique words in corpus: {len(vocab.word_counts):,}")
    print(f"  - Average word frequency: {vocab.total_words / len(vocab.word_counts):.1f}")
    print()
    
    print(f"Vocabulary Configuration:")
    print(f"  - Minimum frequency: {vocab.min_freq}")
    print(f"  - Maximum size: {vocab.max_size or 'unlimited'}")
    print(f"  - Special tokens: {vocab.special_tokens}")
    print()
    
    print(f"Final Vocabulary:")
    print(f"  - Vocabulary size: {vocab.vocab_size:,}")
    print(f"  - Regular words: {vocab.vocab_size - len(vocab.special_tokens):,}")
    print(f"  - Coverage: {(vocab.vocab_size / len(vocab.word_counts)) * 100:.1f}% of unique words")
    
    # Calculate token coverage
    vocab_word_count = sum(vocab.word_freq[word] for word in vocab.word_freq if word != '<UNK>')
    token_coverage = (vocab_word_count / vocab.total_words) * 100
    print(f"  - Token coverage: {token_coverage:.1f}% of all tokens")
    
    print("-" * 60)


def print_frequency_distribution(vocab: SimpleVocabulary, top_n: int = 20) -> None:
    """
    Show the frequency distribution of words in vocabulary.
    
    Args:
        vocab: SimpleVocabulary instance
        top_n: Number of top frequent words to show
    """
    if not vocab.is_built:
        print("Vocabulary not built yet!")
        return
    
    print(f"\n{'='*60}")
    print(f"WORD FREQUENCY DISTRIBUTION")
    print(f"{'='*60}")
    
    # Get words sorted by frequency (excluding special tokens)
    regular_words = [(word, freq) for word, freq in vocab.word_freq.items() 
                    if word not in vocab.special_tokens]
    regular_words.sort(key=lambda x: x[1], reverse=True)
    
    print(f"Top {min(top_n, len(regular_words))} most frequent words:")
    print("-" * 60)
    print(f"{'Rank':<6} {'Word':<20} {'Frequency':<12} {'Percentage':<10}")
    print("-" * 60)
    
    for i, (word, freq) in enumerate(regular_words[:top_n]):
        percentage = (freq / vocab.total_words) * 100
        print(f"{i+1:<6} {word:<20} {freq:<12,} {percentage:<10.2f}%")
    
    # Show frequency ranges
    print(f"\nFrequency Distribution Ranges:")
    print("-" * 60)
    
    frequencies = [freq for _, freq in regular_words]
    ranges = [
        ("Very High (>1000)", len([f for f in frequencies if f > 1000])),
        ("High (100-1000)", len([f for f in frequencies if 100 <= f <= 1000])),
        ("Medium (10-99)", len([f for f in frequencies if 10 <= f <= 99])),
        ("Low (5-9)", len([f for f in frequencies if 5 <= f <= 9])),
        ("Very Low (<5)", len([f for f in frequencies if f < 5]))
    ]
    
    for range_name, count in ranges:
        if count > 0:
            percentage = (count / len(regular_words)) * 100
            print(f"{range_name:<20} {count:<8,} words ({percentage:.1f}%)")
    
    print("-" * 60)


def print_vocabulary_samples(vocab: SimpleVocabulary, sample_size: int = 30) -> None:
    """
    Show sample words from different frequency ranges.
    
    Args:
        vocab: SimpleVocabulary instance
        sample_size: Number of words to sample from each range
    """
    if not vocab.is_built:
        print("Vocabulary not built yet!")
        return
    
    print(f"\n{'='*60}")
    print(f"VOCABULARY WORD SAMPLES")
    print(f"{'='*60}")
    
    # Get words by frequency ranges
    regular_words = [(word, freq) for word, freq in vocab.word_freq.items() 
                    if word not in vocab.special_tokens]
    regular_words.sort(key=lambda x: x[1], reverse=True)
    
    # Define frequency ranges
    ranges = [
        ("High Frequency (top 20)", regular_words[:20]),
        ("Medium Frequency (middle range)", regular_words[len(regular_words)//3:2*len(regular_words)//3]),
        ("Low Frequency (near threshold)", regular_words[-50:])
    ]
    
    for range_name, words_in_range in ranges:
        if not words_in_range:
            continue
            
        print(f"\n{range_name}:")
        print("-" * 40)
        
        sample_words = words_in_range[:sample_size]
        
        # Print in columns for readability
        for i in range(0, len(sample_words), 3):
            row_words = sample_words[i:i+3]
            formatted_words = []
            for word, freq in row_words:
                formatted_words.append(f"{word}({freq})")
            print("  " + "  ".join(f"{w:<20}" for w in formatted_words))
    
    print("-" * 60)


def print_word_lookup_demo(vocab: SimpleVocabulary, demo_words: List[str]) -> None:
    """
    Demonstrate word-to-index and index-to-word lookups.
    
    Args:
        vocab: SimpleVocabulary instance
        demo_words: List of words to demonstrate lookups
    """
    if not vocab.is_built:
        print("Vocabulary not built yet!")
        return
    
    print(f"\n{'='*60}")
    print(f"WORD LOOKUP DEMONSTRATION")
    print(f"{'='*60}")
    
    print("Word → Index lookups:")
    print("-" * 40)
    print(f"{'Word':<15} {'Index':<8} {'In Vocab?':<10}")
    print("-" * 40)
    
    for word in demo_words:
        word_id = vocab.get_word_id(word)
        in_vocab = word in vocab.word2idx
        status = "Yes" if in_vocab else "No (→ UNK)"
        print(f"{word:<15} {word_id:<8} {status:<10}")
    
    print(f"\nIndex → Word lookups (first 10 indices):")
    print("-" * 40)
    print(f"{'Index':<8} {'Word':<15} {'Frequency':<10}")
    print("-" * 40)
    
    for idx in range(min(10, vocab.vocab_size)):
        word = vocab.get_word(idx)
        freq = vocab.word_freq.get(word, 0)
        print(f"{idx:<8} {word:<15} {freq:<10}")
    
    print("-" * 60)


def demonstrate_filtering_effects(vocab: SimpleVocabulary) -> None:
    """
    Show the effects of frequency filtering on vocabulary.
    
    Args:
        vocab: SimpleVocabulary instance
    """
    if not vocab.is_built:
        print("Vocabulary not built yet!")
        return
    
    print(f"\n{'='*60}")
    print(f"FILTERING EFFECTS ANALYSIS")
    print(f"{'='*60}")
    
    # Words that were filtered out
    filtered_out = [word for word, count in vocab.word_counts.items() 
                   if count < vocab.min_freq]
    
    print(f"Words filtered out by frequency threshold:")
    print(f"  - Threshold: {vocab.min_freq}")
    print(f"  - Words removed: {len(filtered_out):,}")
    print(f"  - Percentage removed: {(len(filtered_out) / len(vocab.word_counts)) * 100:.1f}%")
    
    if filtered_out:
        print(f"\nExamples of filtered words (first 20):")
        print("-" * 40)
        sample_filtered = sorted(filtered_out, key=lambda w: vocab.word_counts[w], reverse=True)[:20]
        
        for i in range(0, len(sample_filtered), 4):
            row_words = sample_filtered[i:i+4]
            formatted = [f"{word}({vocab.word_counts[word]})" for word in row_words]
            print("  " + "  ".join(f"{w:<15}" for w in formatted))
    
    # Token impact
    removed_tokens = sum(vocab.word_counts[word] for word in filtered_out)
    print(f"\nImpact on corpus:")
    print(f"  - Tokens from removed words: {removed_tokens:,}")
    print(f"  - Percentage of corpus: {(removed_tokens / vocab.total_words) * 100:.1f}%")
    print(f"  - These will become <UNK> tokens")
    
    print("-" * 60)


# =============================================================================
# EXAMPLE USAGE AND TESTING
# =============================================================================

if __name__ == "__main__":
    # Example usage with sample data
    
    # Sample tokenized sentences for testing
    sample_tokenized_sentences = [
        ['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog'],
        ['word2vec', 'learns', 'word', 'embeddings', 'from', 'large', 'text', 'corpora'],
        ['natural', 'language', 'processing', 'is', 'a', 'fascinating', 'field'],
        ['machine', 'learning', 'algorithms', 'can', 'understand', 'text', 'patterns'],
        ['this', 'is', 'a', 'simple', 'example', 'for', 'educational', 'purposes'],
        ['the', 'word', 'the', 'appears', 'very', 'frequently', 'in', 'text'],
        ['word', 'frequencies', 'follow', 'a', 'power', 'law', 'distribution'],
        ['some', 'words', 'are', 'very', 'common', 'while', 'others', 'are', 'rare']
    ]
    
    print("Testing SimpleVocabulary with sample data...")
    
    # Test with different frequency thresholds
    for min_freq in [1, 2]:
        print(f"\n{'#'*80}")
        print(f"TESTING WITH MIN_FREQ = {min_freq}")
        print(f"{'#'*80}")
        
        # Initialize vocabulary
        vocab = SimpleVocabulary(min_freq=min_freq, max_size=None)
        
        # Build vocabulary
        vocab.build_from_sentences(sample_tokenized_sentences)
        
        # Show all statistics
        print_vocabulary_stats(vocab)
        print_frequency_distribution(vocab, top_n=10)
        print_vocabulary_samples(vocab, sample_size=15)
        
        # Demo word lookups
        demo_words = ['the', 'word', 'rare_word', 'unknown', '<UNK>']
        print_word_lookup_demo(vocab, demo_words)
        
        # Show filtering effects
        demonstrate_filtering_effects(vocab)
    
    print(f"\n{'='*60}")
    print("SimpleVocabulary testing complete!")
    print(f"{'='*60}")