import re
from typing import List
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_STEMMER = PorterStemmer()

# FEVER and Wikipedia text artifact mappings
_FEVER_REPLACEMENTS = {
    "-LRB-": "(",
    "-RRB-": ")",
    "-LSB-": "[",
    "-RSB-": "]",
    "-COLON-": ":",
}

def clean_text(text: str) -> str:
    """Normalize whitespace, quotes, and FEVER wiki artifacts."""
    if not text:
        return ""
    text = text.replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    for k, v in _FEVER_REPLACEMENTS.items():
        text = text.replace(k, v)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess_text(text: str, stem: bool = True, remove_stopwords: bool = True) -> List[str]:
    """
    Standardized Text Preprocessing Pipeline:
    1. Text Cleaning
    2. Tokenization (alphanumeric words)
    3. Stop-word removal (English stop words)
    4. Stemming (Porter Stemmer)
    """
    cleaned = clean_text(text).lower()
    tokens = re.findall(r"\b[a-z0-9]+\b", cleaned)
    
    if remove_stopwords:
        tokens = [t for t in tokens if t not in ENGLISH_STOP_WORDS]
        
    if stem:
        tokens = [_STEMMER.stem(t) for t in tokens]
        
    return tokens

def preprocess_for_tfidf(text: str) -> str:
    """Preprocess text and return as space-separated stemmed tokens for TfidfVectorizer."""
    tokens = preprocess_text(text, stem=True, remove_stopwords=True)
    return " ".join(tokens)

