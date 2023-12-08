import fitz
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
# def importFile(file_name, page_num_down = None, page_num_up = None):
#     #TODO: add support for extracting range of pages (currenlty extracts whole doc)
#     with fitz.open(report_path) as pdf:
#         # Extract text from the first page
#         if page_num_down is None:
#             page_num_down = 0
#         if page_num_up is None:
#             page_num_up = doc.page_count
#         text = ""
#         text = ""
#         for i in range(page_num_down,page_num_up):
#             page = doc[i]
#             text_ = page.get_text("text")
#             text += text_
#     return text

def importFileFromStream(stream, page_num_down = None, page_num_up = None):
    #doc = fitz.Document(stream=stream)
    #page_count = doc.page_count
    #st.session_state['report_page_count'] = page_count
    text = stream
    sentences = []
    page_numbers = []


    sentences_page = sent_tokenize(text)
    for sentence in sentences_page:
        sentences.append(sentence)
        page_numbers.append(1)

    st.session_state['sentences_df'] = pd.DataFrame(data={'ID' : range(0, len(sentences)), 'Original Sentence' : sentences, 'Preprocessed Sentence' : sentences, 'PDF Page Number':page_numbers})
    #print(len(text))
    return text

# def importFileFromStream(stream, page_num_down = None, page_num_up = None):
#     doc = fitz.Document(stream=stream)
#     page_count = doc.page_count
#     st.session_state['report_page_count'] = page_count
#     text = ""
#     n = 0 # artificially created limit, to remove when we will want to process full pdf
#     if page_num_down is None:
#         page_num_down = 0
#     if page_num_up is None:
#         page_num_up = page_count
#     text = ""
#     sentences = []
#     page_numbers = []
#     for i in range(page_num_down,page_num_up):
#         page = doc[i]
#         text_ = page.get_text("text")
#         text += text_
#         n = n + 1
#         sentences_page = sent_tokenize(text_)
#         for sentence in sentences_page:
#             sentences.append(sentence)
#             page_numbers.append(i+1)
#
#     st.session_state['sentences_df'] = pd.DataFrame(data={'ID' : range(0, len(sentences)), 'Original Sentence' : sentences, 'Preprocessed Sentence' : sentences, 'PDF Page Number':page_numbers})
#     #print(len(text))
#     return text


def splitTextIntoSentences(text):
    sentences = text.split(". ")
    clean_sentences = [s.lower() for s in sentences]

def RemoveStopWords(sentences_df, current_directory):
    standard_stopwords = stopwords.words('english')
    sentences = sentences_df['Tokenized Sentence']

    # Check if the file exists
    combined_stopword = standard_stopwords
    if 'custom_stopwords' in st.session_state:
        combined_stopword = combined_stopword + st.session_state['custom_stopwords']
    #try:
    with open("/thesis_project/scripts/custom_stopwords.txt", 'r') as file:
        lines = file.readlines()
        lines = [line.strip() for line in lines]
        #print("custom_stopwords", lines)
        combined_stopword = combined_stopword + lines
        print(combined_stopword)
    # except FileNotFoundError:
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

def GetWordsNumber(column):
    flattened_list = [word for row in column for word in row]
    word_num = len(flattened_list)
    return word_num

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

def TMForDocument(tf_matrix, words_num = 15):

    # Sum term frequencies for each word to find the total frequencies
    word_totals = tf_matrix.sum().sort_values(ascending=False)

    # Select the top 15 words
    top_words = word_totals.head(words_num)

    # Create a DataFrame for the top words
    #top_words_df = pd.DataFrame(top_words).reset_index()
    #top_words_df.columns = ['Word', 'Frequency']

    return top_words


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

def PersonalizeTextRank(sentences_df, stemmer, increase_factor=5):
    keywords = st.session_state['custom_keywords']
    print(keywords)
    stemmed_keywords = [stemmer.stem(word) for word in keywords]
    def count_keywords(sentence):
        return sum(word in stemmed_keywords for word in sentence)
    sentences_df["Keyword Count"] = sentences_df['Stemmed Words'].apply(count_keywords)

    #increase_factor = 5
    personalization_vector = {}
    for _, row in sentences_df.iterrows():
        node_id = row['ID']
        keyword_count = row['Keyword Count']
        importance = increase_factor ** keyword_count
        personalization_vector[node_id] = importance

    # Normalize the personalization vector
    total_importance = sum(personalization_vector.values())
    for node in personalization_vector:
        personalization_vector[node] /= total_importance
    return personalization_vector

def print_Text_Rank_as_text(sentences_df, column_name="Original Sentence"):
    # Flatten the column values
    flattened_values = sentences_df[column_name].explode()

    # Convert the flattened values to a string
    text_output = ' '.join(map(str, flattened_values))

    return text_output

def step_by_step(text):
    sentences_df = st.session_state["sentences_df"].copy()
    sentences_df['Preprocessed Sentence'] = sentences_df['Preprocessed Sentence'].apply(CleanText)

    #we add each sentence as a version of word tokens
    sentences_df['Tokenized Sentence'] = sentences_df['Preprocessed Sentence'].apply(word_tokenize)
    st.session_state['Word_count'] = GetWordsNumber(sentences_df['Tokenized Sentence'])

    #removing combination of general and custom stopwords
    sentences_df = RemoveStopWords(sentences_df, '.')

    #perform stemming in english words
    stemmer = SnowballStemmer("english")
    #sentences_df['Tokenized Sentence'] = sentences_df['Tokenized Sentence'].apply(StringToList)
    sentences_df['Stemmed Words'] = sentences_df['Tokenized Sentence'].apply(lambda x : StemWords(x,stemmer))

    original_count = len(sentences_df)
    unique_words = GetUniqueTokens(sentences_df['Stemmed Words'])
    sentences_df = sentences_df[sentences_df['Stemmed Words'].apply(lambda x: len(x) >= 7)]
    sentences_df.reset_index(drop=True, inplace=True)
    st.session_state['preprocessed_df'] = sentences_df

    removed_count = original_count - len(sentences_df)
    print(removed_count)

    #creating list of unique words
    unique_words = GetUniqueTokens(sentences_df['Stemmed Words'])
    tf = pd.DataFrame(0, index=range(len(sentences_df)), columns=unique_words)
    tf = CreateTermFrequencyMatrix(sentences_df, unique_words, tf)

#list of most occuring words in whole document
    st.session_state['tf_wordcloud'] = TMForDocument(tf)

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

    #st.session_state['similarity_matrix'] = similarity_matrix
    nx_graph = nx.from_numpy_array(similarity_matrix)
    #print(nx_graph)
    #baisc cae without personalization
    if 'custom_keywords' in st.session_state:
        # print("personalized summary")
        # ai_nodes = [1,2,3,7, 22, 28]
        # total_nodes = len(nx_graph.nodes())
        # base_importance = 1 / total_nodes
        # increase_factor = 5
        # personalization_vector = {node: base_importance for node in nx_graph.nodes()}
        # # Increase importance for AI nodes
        # for ai_node in ai_nodes:
        #     personalization_vector[ai_node] *= increase_factor
        #     # Normalize the personalization vector
        # total_importance = sum(personalization_vector.values())
        # print(total_importance)
        # print(personalization_vector)
        # for node in personalization_vector:
        #     personalization_vector[node] /= total_importance
        # print(personalization_vector)
        # total_importance = sum(personalization_vector.values())
        # print(personalization_vector)
        personalization_vector = PersonalizeTextRank(sentences_df, stemmer)
        scores = nx.pagerank(nx_graph, max_iter=2000, alpha=0.9, tol=1.6e-6, personalization=personalization_vector)
    else:
        scores = nx.pagerank(nx_graph, max_iter=2000, alpha=0.9, tol=1.6e-6)



    # Draw the graph
    #plt.figure(figsize=(10, 10))  # Set the size of the figure (optional)
    #nx.draw(nx_graph, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10)
    #plt.title("Graph Representation")

    # Show the graph
    #plt.show()

    # Save the graph to a file
    #plt.savefig("/thesis_project/examples/my_graph.png", format="PNG")
    #sentences_df["Ranks"] = scores
    sentences_df.loc[:, "Ranks"] = scores
    sorted_df = sentences_df.sort_values(by='Ranks', ascending=False)
  # Select the top 20 sentences
    #top_sentences = sorted_df.head(st.session_state['num_sentences'])

    return sorted_df


