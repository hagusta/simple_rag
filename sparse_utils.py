"""
Sparse vector utilities for BM25 implementation in hybrid search.
"""
import re
from typing import List, Dict, Any, Tuple, Optional
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize

# Download NLTK punkt tokenizer if not available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


class BM25Encoder:
    """BM25 sparse vector encoder for text chunks."""

    def __init__(self):
        self.corpus: List[str] = []
        self.bm25_model: Optional[BM25Okapi] = None
        self.vocab: Dict[str, int] = {}
        self.inverse_vocab: Dict[int, str] = {}

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words, removing punctuation and converting to lowercase."""
        # Remove punctuation and convert to lowercase
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        # Tokenize
        tokens = word_tokenize(text)
        # Remove empty tokens and single characters
        tokens = [token for token in tokens if len(token) > 1]
        return tokens

    def fit(self, corpus: List[str]) -> None:
        """Fit the BM25 model on the corpus."""
        self.corpus = corpus
        tokenized_corpus = [self._tokenize(doc) for doc in corpus]

        # Build vocabulary
        all_tokens = set()
        for tokens in tokenized_corpus:
            all_tokens.update(tokens)

        self.vocab = {token: idx for idx, token in enumerate(sorted(all_tokens))}
        self.inverse_vocab = {idx: token for token, idx in self.vocab.items()}

        # Create BM25 model
        self.bm25_model = BM25Okapi(tokenized_corpus)

    def encode(self, query: str) -> Dict[int, float]:
        """Encode a query into sparse vector format."""
        if self.bm25_model is None:
            raise ValueError("BM25 model not fitted. Call fit() first.")

        tokens = self._tokenize(query)
        # Get BM25 scores for the query
        scores = self.bm25_model.get_scores(tokens)

        # Create sparse vector: {token_id: score}
        sparse_vector = {}
        for token, score in zip(tokens, scores):
            if token in self.vocab and score > 0:
                token_id = self.vocab[token]
                sparse_vector[token_id] = float(score)

        return sparse_vector

    def encode_document(self, document: str) -> Dict[int, float]:
        """Encode a document for indexing (returns term frequencies)."""
        tokens = self._tokenize(document)
        token_counts = {}

        for token in tokens:
            if token in self.vocab:
                token_id = self.vocab[token]
                token_counts[token_id] = token_counts.get(token_id, 0) + 1

        return token_counts

    def get_vocab_size(self) -> int:
        """Get the vocabulary size."""
        return len(self.vocab)

    def save_vocab(self, filepath: str) -> None:
        """Save vocabulary and corpus to file."""
        import json
        with open(filepath, 'w') as f:
            json.dump({
                'vocab': self.vocab,
                'corpus': self.corpus,
                'corpus_size': len(self.corpus)
            }, f, indent=2)

    def load_vocab(self, filepath: str) -> None:
        """Load vocabulary and corpus from file, then fit BM25 model."""
        import json
        with open(filepath, 'r') as f:
            data = json.load(f)
            self.vocab = data['vocab']
            # Ensure indices are integers
            self.vocab = {token: int(idx) for token, idx in self.vocab.items()}
            self.inverse_vocab = {idx: token for token, idx in self.vocab.items()}

            if 'corpus' in data:
                self.corpus = data['corpus']
                # Fit the BM25 model with the loaded corpus
                tokenized_corpus = [self._tokenize(doc) for doc in self.corpus]
                self.bm25_model = BM25Okapi(tokenized_corpus)
            else:
                # Backward compatibility: vocab file without corpus
                # Model will remain None, encoding will fail
                self.corpus = []
