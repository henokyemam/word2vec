"""
Simple Word2Vec Data Processing - Educational Implementation
Step 1: Text Processing and Sample Data Loading
"""

import re
from typing import List, Union
from pathlib import Path


class SimpleTextProcessor:
    """
    Simple text preprocessing for Word2Vec.
    Focuses on basic tokenization with clear, educational code.
    """
    
    def __init__(self, lowercase: bool = True, remove_punctuation: bool = True):
        """
        Initialize text processor with basic options.
        
        Args:
            lowercase: Convert text to lowercase
            remove_punctuation: Remove punctuation marks
        """
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize a single line of text into words.
        
        Args:
            text: Input text string
            
        Returns:
            List of tokens (words)
        """
        if not text or not isinstance(text, str):
            return []
        
        # Step 1: Convert to lowercase if requested
        if self.lowercase:
            text = text.lower()
        
        # Step 2: Remove punctuation if requested
        if self.remove_punctuation:
            # Keep only alphanumeric characters and spaces
            text = re.sub(r'[^\w\s]', ' ', text)
        
        # Step 3: Split on whitespace and filter empty strings
        tokens = [token for token in text.split() if token.strip()]
        
        return tokens
    
    def process_sentences(self, sentences: List[str]) -> List[List[str]]:
        """
        Process a list of sentence strings into tokenized sentences.
        
        Args:
            sentences: List of sentence strings
            
        Returns:
            List of tokenized sentences (list of word lists)
        """
        tokenized_sentences = []
        
        for sentence in sentences:
            tokens = self.tokenize(sentence)
            if tokens:  # Only keep non-empty sentences
                tokenized_sentences.append(tokens)
        
        return tokenized_sentences


# =============================================================================
# UTILITY FUNCTIONS FOR LOADING AND VISUALIZING DATA
# =============================================================================

def load_sample_sentences(filepath: Union[str, Path], num_sentences: int = 10) -> List[str]:
    """
    Load sample sentences from a text file for demonstration.
    
    Args:
        filepath: Path to text file
        num_sentences: Number of sentences to load
        
    Returns:
        List of sentence strings
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    sentences = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:  # Skip empty lines
                sentences.append(line)
                if len(sentences) >= num_sentences:
                    break
    
    print(f"Loaded {len(sentences)} sentences from {filepath}")
    return sentences


def print_sample_sentences(sentences: List[str], max_display: int = 5) -> None:
    """
    Display sample sentences in a readable format.
    
    Args:
        sentences: List of sentence strings
        max_display: Maximum number of sentences to display
    """
    print(f"\n{'='*60}")
    print(f"SAMPLE SENTENCES")
    print(f"{'='*60}")
    print(f"Total sentences loaded: {len(sentences)}")
    print(f"Displaying first {min(max_display, len(sentences))} sentences:")
    print("-" * 60)
    
    for i, sentence in enumerate(sentences[:max_display]):
        print(f"{i+1:2d}. {sentence}")
    
    if len(sentences) > max_display:
        print(f"... and {len(sentences) - max_display} more sentences")
    
    print("-" * 60)


def print_tokenization_demo(processor: SimpleTextProcessor, sentences: List[str], max_demo: int = 3) -> List[List[str]]:
    """
    Demonstrate the tokenization process step by step.
    
    Args:
        processor: SimpleTextProcessor instance
        sentences: List of sentence strings
        max_demo: Maximum number of sentences to demonstrate
        
    Returns:
        List of tokenized sentences
    """
    print(f"\n{'='*60}")
    print(f"TOKENIZATION DEMONSTRATION")
    print(f"{'='*60}")
    print(f"Processor settings:")
    print(f"  - Lowercase: {processor.lowercase}")
    print(f"  - Remove punctuation: {processor.remove_punctuation}")
    print("-" * 60)
    
    tokenized_sentences = []
    
    for i, sentence in enumerate(sentences[:max_demo]):
        print(f"\nSentence {i+1}:")
        print(f"  Original: '{sentence}'")
        
        tokens = processor.tokenize(sentence)
        tokenized_sentences.append(tokens)
        
        print(f"  Tokens:   {tokens}")
        print(f"  Count:    {len(tokens)} words")
    
    # Process all remaining sentences without displaying
    for sentence in sentences[max_demo:]:
        tokens = processor.tokenize(sentence)
        if tokens:
            tokenized_sentences.append(tokens)
    
    print(f"\n" + "-" * 60)
    print(f"Total tokenized sentences: {len(tokenized_sentences)}")
    
    return tokenized_sentences


def print_tokenization_stats(tokenized_sentences: List[List[str]]) -> None:
    """
    Print statistics about the tokenized sentences.
    
    Args:
        tokenized_sentences: List of tokenized sentences
    """
    if not tokenized_sentences:
        print("No sentences to analyze!")
        return
    
    # Calculate statistics
    sentence_lengths = [len(sentence) for sentence in tokenized_sentences]
    total_words = sum(sentence_lengths)
    avg_length = total_words / len(tokenized_sentences)
    min_length = min(sentence_lengths)
    max_length = max(sentence_lengths)
    
    # Count unique words
    all_words = []
    for sentence in tokenized_sentences:
        all_words.extend(sentence)
    unique_words = set(all_words)
    
    print(f"\n{'='*60}")
    print(f"TOKENIZATION STATISTICS")
    print(f"{'='*60}")
    print(f"Sentences:        {len(tokenized_sentences)}")
    print(f"Total words:      {total_words}")
    print(f"Unique words:     {len(unique_words)}")
    print(f"Average length:   {avg_length:.1f} words/sentence")
    print(f"Min length:       {min_length} words")
    print(f"Max length:       {max_length} words")
    print("-" * 60)


def print_word_examples(tokenized_sentences: List[List[str]], num_examples: int = 20) -> None:
    """
    Print examples of words found in the tokenized sentences.
    
    Args:
        tokenized_sentences: List of tokenized sentences  
        num_examples: Number of example words to show
    """
    # Collect all words
    all_words = []
    for sentence in tokenized_sentences:
        all_words.extend(sentence)
    
    # Get unique words
    unique_words = list(set(all_words))
    unique_words.sort()  # Sort alphabetically
    
    print(f"\n{'='*60}")
    print(f"WORD EXAMPLES")
    print(f"{'='*60}")
    print(f"First {min(num_examples, len(unique_words))} unique words (alphabetically):")
    print("-" * 60)
    
    # Print words in columns for better readability
    words_to_show = unique_words[:num_examples]
    for i in range(0, len(words_to_show), 4):
        row_words = words_to_show[i:i+4]
        print("  ".join(f"{word:<15}" for word in row_words))
    
    if len(unique_words) > num_examples:
        print(f"... and {len(unique_words) - num_examples} more unique words")
    
    print("-" * 60)


# =============================================================================
# EXAMPLE USAGE AND TESTING
# =============================================================================

if __name__ == "__main__":
    # Example usage - you can test with your own text file
    
    # Create sample data for testing if no file is available
    sample_text = [
        "The quick brown fox jumps over the lazy dog.",
        "Word2Vec learns word embeddings from large text corpora.",
        "Natural language processing is a fascinating field!",
        "Machine learning algorithms can understand text patterns.",
        "This is a simple example for educational purposes."
    ]
    
    print("Testing SimpleTextProcessor with sample data...")
    
    # Initialize processor
    processor = SimpleTextProcessor(lowercase=True, remove_punctuation=True)
    
    # Demonstrate the full pipeline
    print_sample_sentences(sample_text)
    tokenized_sentences = print_tokenization_demo(processor, sample_text)
    print_tokenization_stats(tokenized_sentences)
    print_word_examples(tokenized_sentences)
    
    print(f"\n{'='*60}")
    print("SimpleTextProcessor testing complete!")
    print(f"{'='*60}")