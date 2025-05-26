"""
Simple Word2Vec Data Processing - Educational Implementation
Step 3: Subsampling of Frequent Words
"""

import math
import random
from typing import List, Dict, Tuple
from vocab_builder import SimpleVocabulary


class SimpleSubSampler:
    """
    Simple subsampling implementation for Word2Vec.
    Randomly discards frequent words based on their frequency.
    
    Formula from Word2Vec paper:
    P(discard) = 1 - sqrt(threshold / word_frequency_ratio)
    """
    
    def __init__(self, vocab: SimpleVocabulary, threshold: float = 1e-3):
        """
        Initialize subsampler.
        
        Args:
            vocab: SimpleVocabulary instance
            threshold: Subsampling threshold (default 1e-3 from paper)
        """
        if not vocab.is_built:
            raise ValueError("Vocabulary must be built before creating subsampler")
            
        self.vocab = vocab
        self.threshold = threshold
        
        # Calculate discard probabilities for each word
        self.discard_probs = self._calculate_discard_probabilities()
        
        print(f"\n{'='*60}")
        print(f"SUBSAMPLER INITIALIZED")
        print(f"{'='*60}")
        print(f"Threshold: {self.threshold}")
        print(f"Words with discard probability > 0: {sum(1 for p in self.discard_probs.values() if p > 0)}")
        print("-" * 60)
    
    def _calculate_discard_probabilities(self) -> Dict[int, float]:
        """
        Calculate discard probability for each word based on frequency.
        
        Formula: P(discard) = 1 - sqrt(threshold / word_frequency_ratio)
        where word_frequency_ratio = word_count / total_words
        
        Returns:
            Dictionary mapping word_id to discard probability
        """
        discard_probs = {}
        
        for word, word_id in self.vocab.word2idx.items():
            word_count = self.vocab.word_freq[word]
            
            # Skip special tokens
            if word in self.vocab.special_tokens:
                discard_probs[word_id] = 0.0
                continue
            
            # Calculate word frequency ratio
            word_freq_ratio = word_count / self.vocab.total_words
            
            # Apply subsampling formula
            if word_freq_ratio > self.threshold:
                discard_prob = 1 - math.sqrt(self.threshold / word_freq_ratio)
                discard_probs[word_id] = max(0.0, discard_prob)
            else:
                discard_probs[word_id] = 0.0
        
        return discard_probs
    
    def should_discard_word(self, word_id: int) -> bool:
        """
        Determine if a word should be discarded based on subsampling.
        
        Args:
            word_id: Word index
            
        Returns:
            True if word should be discarded, False otherwise
        """
        discard_prob = self.discard_probs.get(word_id, 0.0)
        return random.random() < discard_prob
    
    def subsample_sentence(self, sentence: List[int]) -> List[int]:
        """
        Apply subsampling to a sentence (list of word IDs).
        
        Args:
            sentence: List of word IDs
            
        Returns:
            Subsampled sentence (some words removed)
        """
        subsampled = []
        for word_id in sentence:
            if not self.should_discard_word(word_id):
                subsampled.append(word_id)
        return subsampled
    
    def subsample_sentences(self, sentences: List[List[int]]) -> List[List[int]]:
        """
        Apply subsampling to multiple sentences.
        
        Args:
            sentences: List of sentences (each sentence is list of word IDs)
            
        Returns:
            List of subsampled sentences
        """
        return [self.subsample_sentence(sentence) for sentence in sentences]


# =============================================================================
# UTILITY FUNCTIONS FOR SUBSAMPLING ANALYSIS
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
    for sentence in tokenized_sentences:
        id_sentence = [vocab.get_word_id(word) for word in sentence]
        id_sentences.append(id_sentence)
    return id_sentences


def print_subsampling_probabilities(subsampler: SimpleSubSampler, top_n: int = 20) -> None:
    """
    Display subsampling probabilities for words.
    
    Args:
        subsampler: SimpleSubSampler instance
        top_n: Number of words to display
    """
    print(f"\n{'='*60}")
    print(f"SUBSAMPLING PROBABILITIES")
    print(f"{'='*60}")
    print(f"Threshold: {subsampler.threshold}")
    print(f"Formula: P(discard) = 1 - sqrt(threshold / word_frequency_ratio)")
    print("-" * 60)
    
    # Get words sorted by discard probability (descending)
    word_probs = []
    for word_id, discard_prob in subsampler.discard_probs.items():
        word = subsampler.vocab.get_word(word_id)
        if word not in subsampler.vocab.special_tokens:
            word_count = subsampler.vocab.word_freq[word]
            word_freq_ratio = word_count / subsampler.vocab.total_words
            word_probs.append((word, word_count, word_freq_ratio, discard_prob))
    
    # Sort by discard probability (highest first)
    word_probs.sort(key=lambda x: x[3], reverse=True)
    
    print(f"{'Word':<15} {'Count':<8} {'Freq%':<8} {'Discard%':<10} {'Effect'}")
    print("-" * 60)
    
    for word, count, freq_ratio, discard_prob in word_probs[:top_n]:
        freq_percent = freq_ratio * 100
        discard_percent = discard_prob * 100
        
        if discard_prob > 0.5:
            effect = "High"
        elif discard_prob > 0.1:
            effect = "Medium"
        elif discard_prob > 0:
            effect = "Low"
        else:
            effect = "None"
            
        print(f"{word:<15} {count:<8} {freq_percent:<8.3f} {discard_percent:<10.1f} {effect}")
    
    # Summary statistics
    high_discard = sum(1 for _, _, _, p in word_probs if p > 0.5)
    medium_discard = sum(1 for _, _, _, p in word_probs if 0.1 < p <= 0.5)
    low_discard = sum(1 for _, _, _, p in word_probs if 0 < p <= 0.1)
    no_discard = sum(1 for _, _, _, p in word_probs if p == 0)
    
    print(f"\nSubsampling Effect Summary:")
    print("-" * 40)
    print(f"High discard (>50%):     {high_discard:4d} words")
    print(f"Medium discard (10-50%): {medium_discard:4d} words")
    print(f"Low discard (0-10%):     {low_discard:4d} words")
    print(f"No discard (0%):         {no_discard:4d} words")
    print("-" * 60)


def print_subsampling_demo(subsampler: SimpleSubSampler, 
                          sample_sentences: List[List[str]], 
                          num_demos: int = 3,
                          num_runs: int = 3) -> None:
    """
    Demonstrate subsampling on sample sentences.
    
    Args:
        subsampler: SimpleSubSampler instance
        sample_sentences: List of tokenized sentences
        num_demos: Number of sentences to demonstrate
        num_runs: Number of subsampling runs per sentence
    """
    print(f"\n{'='*60}")
    print(f"SUBSAMPLING DEMONSTRATION")
    print(f"{'='*60}")
    print(f"Showing {num_runs} random subsampling runs for each sentence")
    print("-" * 60)
    
    # Convert sentences to IDs
    id_sentences = convert_sentences_to_ids(sample_sentences, subsampler.vocab)
    
    for i, (sentence_tokens, sentence_ids) in enumerate(zip(sample_sentences[:num_demos], id_sentences[:num_demos])):
        print(f"\nSentence {i+1}:")
        print(f"Original: {' '.join(sentence_tokens)}")
        print(f"Length: {len(sentence_tokens)} words")
        
        # Show which words have discard probabilities
        word_info = []
        for token, word_id in zip(sentence_tokens, sentence_ids):
            discard_prob = subsampler.discard_probs.get(word_id, 0.0)
            word_info.append((token, discard_prob))
        
        print("Word discard probabilities:")
        for token, prob in word_info:
            if prob > 0:
                print(f"  {token}: {prob:.3f}")
            else:
                print(f"  {token}: 0.000")
        
        print(f"\nSubsampling runs:")
        for run in range(num_runs):
            subsampled_ids = subsampler.subsample_sentence(sentence_ids)
            subsampled_tokens = [subsampler.vocab.get_word(word_id) for word_id in subsampled_ids]
            
            removed_count = len(sentence_ids) - len(subsampled_ids)
            removal_rate = (removed_count / len(sentence_ids)) * 100 if sentence_ids else 0
            
            print(f"  Run {run+1}: {' '.join(subsampled_tokens)}")
            print(f"         Length: {len(subsampled_tokens)} (-{removed_count}, {removal_rate:.1f}% removed)")
        
        print("-" * 40)


def analyze_subsampling_effects(subsampler: SimpleSubSampler, 
                              original_sentences: List[List[int]], 
                              num_trials: int = 100) -> None:
    """
    Analyze the statistical effects of subsampling on a corpus.
    
    Args:
        subsampler: SimpleSubSampler instance
        original_sentences: List of sentences with word IDs
        num_trials: Number of subsampling trials for statistics
    """
    print(f"\n{'='*60}")
    print(f"SUBSAMPLING EFFECTS ANALYSIS")
    print(f"{'='*60}")
    print(f"Analyzing effects over {num_trials} random trials")
    print("-" * 60)
    
    # Original statistics
    original_total_words = sum(len(sentence) for sentence in original_sentences)
    original_sentences_count = len(original_sentences)
    
    print(f"Original Corpus:")
    print(f"  - Sentences: {original_sentences_count:,}")
    print(f"  - Total words: {original_total_words:,}")
    print(f"  - Average sentence length: {original_total_words / original_sentences_count:.1f}")
    
    # Run multiple subsampling trials
    total_words_removed = 0
    total_sentences_removed = 0
    word_removal_counts = {}
    
    for trial in range(num_trials):
        subsampled_sentences = subsampler.subsample_sentences(original_sentences)
        
        # Count removals
        trial_words_removed = 0
        trial_sentences_removed = 0
        
        for orig, subsampled in zip(original_sentences, subsampled_sentences):
            words_removed = len(orig) - len(subsampled)
            trial_words_removed += words_removed
            
            if len(subsampled) == 0:
                trial_sentences_removed += 1
            
            # Track which words were removed
            for word_id in orig:
                if word_id not in subsampled:
                    word_removal_counts[word_id] = word_removal_counts.get(word_id, 0) + 1
        
        total_words_removed += trial_words_removed
        total_sentences_removed += trial_sentences_removed
    
    # Calculate averages
    avg_words_removed = total_words_removed / num_trials
    avg_sentences_removed = total_sentences_removed / num_trials
    avg_removal_rate = (avg_words_removed / original_total_words) * 100
    
    print(f"\nSubsampling Results (averaged over {num_trials} trials):")
    print(f"  - Words removed per trial: {avg_words_removed:.1f}")
    print(f"  - Word removal rate: {avg_removal_rate:.1f}%")
    print(f"  - Sentences completely removed: {avg_sentences_removed:.1f}")
    print(f"  - Sentence removal rate: {(avg_sentences_removed / original_sentences_count) * 100:.1f}%")
    
    # Most frequently removed words
    if word_removal_counts:
        print(f"\nMost Frequently Removed Words:")
        print("-" * 40)
        print(f"{'Word':<15} {'Removed':<8} {'Rate%':<8} {'Discard%':<10}")
        print("-" * 40)
        
        sorted_removals = sorted(word_removal_counts.items(), key=lambda x: x[1], reverse=True)
        
        for word_id, removal_count in sorted_removals[:10]:
            word = subsampler.vocab.get_word(word_id)
            removal_rate = (removal_count / num_trials) * 100
            discard_prob = subsampler.discard_probs.get(word_id, 0.0) * 100
            
            print(f"{word:<15} {removal_count:<8} {removal_rate:<8.1f} {discard_prob:<10.1f}")
    
    print("-" * 60)


def compare_with_without_subsampling(vocab: SimpleVocabulary,
                                   sample_sentences: List[List[str]],
                                   threshold: float = 1e-3) -> None:
    """
    Compare corpus statistics with and without subsampling.
    
    Args:
        vocab: SimpleVocabulary instance
        sample_sentences: List of tokenized sentences
        threshold: Subsampling threshold
    """
    print(f"\n{'='*60}")
    print(f"COMPARISON: WITH vs WITHOUT SUBSAMPLING")
    print(f"{'='*60}")
    
    # Convert to IDs
    id_sentences = convert_sentences_to_ids(sample_sentences, vocab)
    
    # Create subsampler
    subsampler = SimpleSubSampler(vocab, threshold)
    
    # Original statistics
    original_total_words = sum(len(sentence) for sentence in id_sentences)
    original_unique_words = len(set(word_id for sentence in id_sentences for word_id in sentence))
    
    # Subsampled statistics (average over multiple runs)
    num_runs = 50
    subsampled_totals = []
    subsampled_uniques = []
    
    for _ in range(num_runs):
        subsampled_sentences = subsampler.subsample_sentences(id_sentences)
        subsampled_total = sum(len(sentence) for sentence in subsampled_sentences)
        subsampled_unique = len(set(word_id for sentence in subsampled_sentences for word_id in sentence if sentence))
        
        subsampled_totals.append(subsampled_total)
        subsampled_uniques.append(subsampled_unique)
    
    avg_subsampled_total = sum(subsampled_totals) / len(subsampled_totals)
    avg_subsampled_unique = sum(subsampled_uniques) / len(subsampled_uniques)
    
    print(f"{'Metric':<25} {'Original':<12} {'Subsampled':<12} {'Change':<12}")
    print("-" * 65)
    print(f"{'Total words':<25} {original_total_words:<12,} {avg_subsampled_total:<12.0f} {((avg_subsampled_total - original_total_words) / original_total_words * 100):+.1f}%")
    print(f"{'Unique words':<25} {original_unique_words:<12,} {avg_subsampled_unique:<12.0f} {((avg_subsampled_unique - original_unique_words) / original_unique_words * 100):+.1f}%")
    print(f"{'Avg sentence length':<25} {original_total_words / len(id_sentences):<12.1f} {avg_subsampled_total / len(id_sentences):<12.1f} {((avg_subsampled_total / len(id_sentences)) - (original_total_words / len(id_sentences))):+.1f}")
    
    print(f"\nSubsampling Impact:")
    word_reduction = ((original_total_words - avg_subsampled_total) / original_total_words) * 100
    print(f"  - Word reduction: {word_reduction:.1f}%")
    print(f"  - Threshold used: {threshold}")
    print(f"  - Primary effect: Reduces frequent words, keeps rare words")
    
    print("-" * 60)


# =============================================================================
# EXAMPLE USAGE AND TESTING
# =============================================================================

if __name__ == "__main__":
    # Import required modules (assuming they exist)
    try:
        from vocab_builder import SimpleVocabulary
    except ImportError:
        print("This module requires SimpleVocabulary. Please ensure it's available.")
        exit(1)
    
    # Sample data for testing
    sample_sentences = [
        ['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog'],
        ['the', 'dog', 'is', 'very', 'lazy', 'and', 'sleeps', 'all', 'day'],
        ['word2vec', 'learns', 'word', 'embeddings', 'from', 'text', 'data'],
        ['the', 'algorithm', 'uses', 'the', 'skip', 'gram', 'model'],
        ['frequent', 'words', 'like', 'the', 'and', 'is', 'appear', 'often'],
        ['subsampling', 'reduces', 'the', 'frequency', 'of', 'common', 'words'],
        ['this', 'helps', 'the', 'model', 'learn', 'better', 'representations'],
        ['the', 'the', 'the', 'appears', 'very', 'frequently', 'in', 'text']
    ]
    
    print("Testing SimpleSubSampler with sample data...")
    
    # Build vocabulary first
    vocab = SimpleVocabulary(min_freq=1, max_size=None)
    vocab.build_from_sentences(sample_sentences)
    
    # Test subsampling with different thresholds
    for threshold in [1e-3, 1e-4, 1e-2]:
        print(f"\n{'#'*80}")
        print(f"TESTING WITH THRESHOLD = {threshold}")
        print(f"{'#'*80}")
        
        # Create subsampler
        subsampler = SimpleSubSampler(vocab, threshold)
        
        # Show probabilities
        print_subsampling_probabilities(subsampler, top_n=15)
        
        # Demonstrate on sample sentences
        print_subsampling_demo(subsampler, sample_sentences, num_demos=3, num_runs=3)
        
        # Analyze effects
        id_sentences = convert_sentences_to_ids(sample_sentences, vocab)
        analyze_subsampling_effects(subsampler, id_sentences, num_trials=20)
        
        # Compare with/without subsampling
        compare_with_without_subsampling(vocab, sample_sentences, threshold)
    
    print(f"\n{'='*60}")
    print("SimpleSubSampler testing complete!")
    print(f"{'='*60}")