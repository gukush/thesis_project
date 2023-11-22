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
    #TODO: add support for extracting range of pages (currenlty extracts whole doc)
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
    n = 0 # artificially created limit, to remove when we will want to process full pdf
    for page in doc:
        text_ = page.get_text("text")
        text += text_
        n = n + 1
        if n > 10:
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

def StemWords(tokens, stemmer):
    # Apply stemming to each token in the list
    stemmed_tokens = [stemmer.stem(token) if token.isalpha() else token for token in tokens]
    return stemmed_tokens

def GetUniqueTokens(column):
    # flatening the column
    flattened_list = [word for row in column for word in row]

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

def step_by_step(text):
    sentences = sent_tokenize(text)
    sentences_df = pd.DataFrame(data={'ID' : range(0, len(sentences)), 'Original Sentence' : sentences, 'Preprocessed Sentence' : sentences})
    sentences_df['Preprocessed Sentence'] = sentences_df['Preprocessed Sentence'].apply(CleanText)
    #we add each sentence as a version of word tokens
    sentences_df['Tokenized Sentence'] = sentences_df['Preprocessed Sentence'].apply(word_tokenize)
    #removing combination of general and custom stopwords
    sentences_df = RemoveStopWords(sentences_df, '.')
    #perform stemming in english words
    stemmer = SnowballStemmer("english")
    #sentences_df['Tokenized Sentence'] = sentences_df['Tokenized Sentence'].apply(StringToList)
    sentences_df['Stemmed Words'] = sentences_df['Tokenized Sentence'].apply(lambda x : StemWords(x,stemmer))
    #creating list of unique words
    unique_words = GetUniqueTokens(sentences_df['Stemmed Words'])
    tf = pd.DataFrame(0, index=range(len(sentences_df)), columns=unique_words)
    tf = CreateTermFrequencyMatrix(sentences_df,unique_words,tf)
#    tf = tf.rename(columns=unique_words)
#
# #checking if we don't have empty vectors
#    non_empty_rows = ~np.all(tf == 0, axis=1)
#
# # Filter the array to keep non-empty rows
#    tf = tf[non_empty_rows]
# #saving tf
# #df = pd.DataFrame(tf)
    #creating similarity matrix
    similarity_matrix = np.zeros((len(sentences_df), len(sentences_df)), dtype = float)
    for i in range(len(sentences_df)):
      for j in range(len(sentences_df)):
          if i != j:
              # Extract the term frequency vectors for the two sentences
              tf_vector_i = tf.iloc[i].values
              tf_vector_j = tf.iloc[j].values
             # Calculate cosine similarity and assign it to the similarity matrix
              similarity_matrix[i, j] = cosine_similarity(tf_vector_i, tf_vector_j)
# df = pd.DataFrame(similarity_matrix)
    #V = textrank(similarity_matrix, 400, 0.85)
    nx_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(nx_graph, max_iter = 1000)
    sentences_df["Ranks"] = scores
    sorted_df = sentences_df.sort_values(by='Ranks', ascending=False)
  # Select the top 20 sentences
    top_20_sentences = sorted_df.head(20)
    return top_20_sentences


