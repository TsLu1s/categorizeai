import streamlit as st

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re 

def is_nltk_resource_downloaded(resource):
    """Check if an NLTK resource is already downloaded."""
    try:
        if resource == 'punkt':
            nltk.data.find('tokenizers/punkt')
        elif resource == 'wordnet':
            nltk.data.find('corpora/wordnet')
        elif resource == 'stopwords':
            nltk.data.find('corpora/stopwords')
        else:
            nltk.data.find(f'tokenizers/{resource}')
        return True
    except LookupError:
        return False

def load_nltk_components():
    """
    Load NLTK components efficiently with support for multiple languages.
    Downloads resources only if necessary.
     
    Returns:
        tuple: (WordNetLemmatizer, dict of language codes to stopword sets)
    """
    
    # Language codes for major languages
    # Note: Some languages might not have stopwords in NLTK
    languages = {
        'en': 'english',
        'it': 'italian',
        'es': 'spanish',
        'pt': 'portuguese',
        'fr': 'french'
    }
    
    # Resources to check and download if needed
    required_resources = ['punkt', 'stopwords', 'wordnet']
    
    # Download missing resources
    for resource in required_resources:
        if not is_nltk_resource_downloaded(resource):
            try:
                nltk.download(resource, quiet=True)
            except Exception as e:
                st.error(f"Failed to download {resource}. Error: {str(e)}")
                raise
    
    # Initialize components
    lemmatizer = WordNetLemmatizer()
    stopwords_dict = {}
    
    # Load stopwords for each language
    with st.spinner("Loading multilingual stopwords..."):
        for lang_code, lang_name in languages.items():
            try:
                stopwords_dict[lang_code] = set(stopwords.words(lang_name))
            except OSError as e:
                st.warning(f"Could not load stopwords for {lang_name}. Error: {str(e)}")
                stopwords_dict[lang_code] = set()  # Empty set as fallback
            except ValueError as e:
                st.warning(f"Language {lang_name} not available in NLTK stopwords")
                stopwords_dict[lang_code] = set()  # Empty set as fallback
    
    return lemmatizer, stopwords_dict
    
def preprocess_text(text, lemmatizer=None, stopwords_dict=None):
    """
    Preprocess text by removing stop words from all supported languages.
    
    Args:
        text (str): Input text to preprocess
        lemmatizer: NLTK lemmatizer instance, optional
        stopwords_dict: Dictionary of language codes to stopword sets, optional
        
    Returns:
        str: Preprocessed text with stop words removed from all languages
    """
    if not text or not isinstance(text, str):
        return ""
        
    # Load components if not provided
    if lemmatizer is None or stopwords_dict is None:
        lemmatizer, stopwords_dict = load_nltk_components()
    
    # Create a combined set of stop words from all languages
    all_stop_words = set().union(*stopwords_dict.values())
    
    # Basic cleaning
    text = text.lower()
    
    # Combined dictionary of patterns and their replacements
    cleaning_patterns = {       
        # Unwanted characters
        re.compile(r'\d+'): ' ',                       # Numbers
        re.compile(r'[^\w\s]'): ' ',                   # Punctuation
        re.compile(r'_{2,}'): ' ',                     # Multiple underscores
        re.compile(r'[\u0080-\uffff]'): ' ',           # Non-ASCII characters
        re.compile(r'[\u0021-\u002F]'): ' ',           # ASCII punctuation marks
        re.compile(r'[\u003A-\u0040]'): ' ',           
        re.compile(r'[\u005B-\u0060]'): ' ',           
        re.compile(r'[\u007B-\u007E]'): ' ',           
        
        # Common artifacts
        re.compile(r'\bbr\b'): ' ',                    # br tags
        re.compile(r'\n'): ' ',                        # Newlines
        re.compile(r'\r'): ' ',                        # Carriage returns
        re.compile(r'\t'): ' ',                        # Tabs
        re.compile(r'\b\w{1,2}\b'): ' ',               # 1-2 character words
        re.compile(r'\s+'): ' ',                       # Multiple spaces
    }
    
    # Apply all patterns in a single loop
    for pattern, replacement in cleaning_patterns.items():
        text = pattern.sub(replacement, text)

    # Tokenization and preprocessing
    words = word_tokenize(text)
    
    # Process words: lemmatize and remove stop words from all languages
    processed_words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in all_stop_words and len(word) > 1
    ]
    
    return ' '.join(processed_words)