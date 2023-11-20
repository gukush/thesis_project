import fitz
import numpy as np
import pandas as pd

import nltk
try: 
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try: 
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

import re
from numpy.linalg import norm
import networkx as nx
import ast

#function responsible for importing given pages
def importFile(page_num_down, page_num_up, file_name):
    with fitz.open(report_path) as pdf:
        # Extract text from the first page
        text = ""
        for page in pdf:
            text_ = page.get_text("text")
            text += text_
    return text

def importFileFromStream(stream):
    doc = fitz.Document(stream=stream)
    text = ""
    for page in doc:
        text_ = page.get_text("text")
        text += text_
        break
    return text

def splitTextIntoSentences(text):
    sentences = text.split(". ")
    clean_sentences = [s.lower() for s in sentences]

def RemoveStopWords(sentences_df, current_directory):
    standard_stopwords = stopwords.words('english')
    sentences = sentences_df['Tokenized Sentence']

    print(sentences)

    # Check if the file exists
    combined_stopword = standard_stopwords
    try:
        with open("./scripts/custom_stopwords.txt", 'r') as file:
            lines = file.readlines()
            lines = [line.strip() for line in lines]
            #print("custom_stopwords", lines)
            combined_stopword = lines
            #print(combined_stopword)
    except FileNotFoundError:
        print(f"The file with custom stopword does not exist.")
        lines = []

    cleaned_sentences = []
    for sentence in sentences:
        # Create a list of words that are not in stopwords
        filtered_sentence = [word for word in sentence if word not in combined_stopword]
        # Append the filtered list of words to 'cleaned_sentences'
        cleaned_sentences.append(filtered_sentence)

    # Assuming 'sentences_df' is a DataFrame and you want to store these lists in a new column
    sentences_df['Tokenized Sentence'] = cleaned_sentences
    return sentences_df

    #clean_sentences = [RemoveStopWords(r.split(), combined_stopword) for r in clean_sentences]

def RemoveShortSentences(sentences_df, sentence_minimal_len):
    sentences = sentences_df["Preprocessed Sentences"]

    filtered_sentences = [i for i in sentences if len(i.split()) >= sentence_minimal_len ]

    return filtered_sentences

    # for i in sentences:
    #     if len(cleaned_sentence.split()) >= sentence_minimal_len:
    #         cleaned_sentences.append(cleaned_sentence)
    #     else:
    #         print(cleaned_sentence)

#function resposible for cleaning
def CleanText(sentence):
    cleaned_text = re.sub(r'[^a-zA-Z]', ' ', sentence).lower()
    return cleaned_text

def StemWords(tokens):
    # Apply stemming to each token in the list
    stemmed_tokens = [stemmer.stem(token) if token.isalpha() else token for token in tokens]
    return stemmed_tokens

def GetUniqueTokens(column):
    # flatening the column
    flattened_list = [word for row in sentences_df['Stemmed Words'] for word in row]

    # Get unique values in for of lists
    unique_words = set(flattened_list)
    unique_words_list = list(unique_words)
    return unique_words_list

min_threhold = 1e-5
def textrank(M, iteration, D):
    i = 0
    N = len(M)
    V = [1 / N] * N
    V = V / np.linalg.norm(V, 1)
    Prev_V = 0
    M_T = (D * M + (1 - D) / N)
    while i < iteration:
        V = np.dot(M_T, V)
        if abs(Prev_V - sum(V)) < min_threhold:
            break
        else:
            Prev_V = sum(V)
        i += 1
        print(i)
    return V

def pagerank(M, num_iterations: int = 100, d: float = 0.85):
    """PageRank algorithm with explicit number of iterations. Returns ranking of nodes (pages) in the adjacency matrix.

    Parameters
    ----------
    M : numpy array
        adjacency matrix where M_i,j represents the link from 'j' to 'i', such that for all 'j'
        sum(i, M_i,j) = 1
    num_iterations : int, optional
        number of iterations, by default 100
    d : float, optional
        damping factor, by default 0.85

    Returns
    -------
    numpy array
        a vector of ranks such that v_i is the i-th rank from [0, 1],
        v sums to 1

    """
    N = M.shape[1]
    v = np.ones(N) / N
    M_hat = (d * M + (1 - d) / N)
    for i in range(num_iterations):
        v = M_hat @ v
    return v

def cosine_similarity(a,b):
    cosine = np.dot(a, b) / (norm(a) * norm(b))
    return cosine

def is_sentence_long_enough(sentence, min_length):
    return len(sentence.split()) >= min_length

def RemoveShortSentences(sentences_df, sentence_minimal_len):
    # Apply the is_sentence_long_enough function to filter the dataframe
    filtered_df = sentences_df[sentences_df["Preprocessed Sentences"].apply(is_sentence_long_enough, min_length=sentence_minimal_len)]
    return filtered_df

#sometimes we need to change string for list
def StringToList(string):
    try:
        # Assuming the input is like your example, replace elements without quotes
        formatted_string = string.replace("[", '["').replace("]", '"]').replace(", ", '", "')
        return ast.literal_eval(formatted_string)
    except (ValueError, SyntaxError):
        # Handle the case where the string is not a valid list representation
        return []

def CreateTermFrequencyMatrix(sentences_df, unique_words, tf):
    for row in range(len(sentences_df)):
        # Iterate over each unique word
        for word in unique_words:
            # Count the occurrences of the word in the sentence
            tf.at[row, word] = sentences_df.at[row, 'Stemmed Words'].count(word)
    return tf

def extract_numerics_with_context(text, num_words_before=1):
    """
    Extract numeric values and a specified number of preceding words from a text.

    Parameters:
    text (str): The text to extract information from.
    num_words_before (int): The number of words before the numeric value to extract.

    Returns:
    list of tuples: A list where each tuple contains the numeric value and its preceding words.
    """
    # Tokenize the text
    tokens = word_tokenize(text)

    # Regular expression for finding numbers
    pattern = r'\d+'

    # Find all matches and capture preceding words
    results = []
    matches = re.finditer(pattern, text)
    for match in matches:
        start = match.start()
        token_index = len(word_tokenize(text[:start]))
        context_start = max(0, token_index - num_words_before)
        context_words = ' '.join(tokens[context_start:token_index])
        number = match.group()
        results.append((number, context_words))

    return results
