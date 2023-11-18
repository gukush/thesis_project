import pdfplumber
import numpy as np
import pandas as pd

import nltk
#nltk.download('punkt')
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

import re
from numpy.linalg import norm
import networkx as nx
import ast

#function responsible for importing given pages
def importFile(page_num_down, page_num_up, file_name):
    with pdfplumber.open(report_path) as pdf:
        # Extract text from the first page
        text = ""
        for i in range(page_num_down, page_num_up):
            first_page = pdf.pages[i]
            text_ = first_page.extract_text()
            text += text_
    return text

def splitTextIntoSentences(text):
    sentences = text.split(". ")
    clean_sentences = [s.lower() for s in sentences]

def RemoveStopWords(sentences_df, current_directory):
    standard_stopwords = stopwords.words('english')
    sentences = sentences_df['Tokenized Sentence']

    print(sentences)

    # Check if the file exists
    try:
        with open(current_directory + "custom_stopwords.txt", 'r') as file:
            lines = file.readlines()
            lines = [line.strip() for line in lines]
            #print("custom_stopwords", lines)
            combined_stopword = standard_stopwords + lines
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
#def get_companies_name(file_path)

#report_path = current_directory + "Hestia_sfcr.pdf"
current_directory = "../examples/"

#remove_short_sentences = False
sentence_minimal_len = 5
sentence_to_print = 10  # Change N to the number of top sentences you want to print

page_num_down = 0
page_num_up = 50
report_path = current_directory + "shell-sustainability-report-2022.pdf" #"apple_2022.pdf"

# with pdfplumber.open(report_path) as pdf:
#     # Extract text from the first page
#     text = ""
#     for i in range(page_num_down, page_num_up):
#         first_page = pdf.pages[i]
#         text_ = first_page.extract_text()
#         text += text_

#import text from file
text = importFile(page_num_down, page_num_up, report_path)
#print(text)

#split text into sentences
sentences = sent_tokenize(text)

#create data frame of sentences
#we add ID's to identify the order of sentences
sentences_df = pd.DataFrame(data={'ID' : range(0, len(sentences)), 'Original Sentence' : sentences, 'Preprocessed Sentence' : sentences})

#preprocessing

sentences_df['Preprocessed Sentence'] = sentences_df['Preprocessed Sentence'].apply(CleanText)

#we add each sentence as a version of word tokens
sentences_df['Tokenized Sentence'] = sentences_df['Preprocessed Sentence'].apply(word_tokenize)

#removing combination of general and custom stopwords
sentences_df = RemoveStopWords(sentences_df, current_directory)

#perform stemming in english words
stemmer = SnowballStemmer("english")
#sentences_df['Tokenized Sentence'] = sentences_df['Tokenized Sentence'].apply(StringToList)
sentences_df['Stemmed Words'] = sentences_df['Tokenized Sentence'].apply(StemWords)

# Stemming and removing extra spaces
# cleaned_sentences_stemmed = []
# for sentence in cleaned_sentences:
#     # Stem each word in the sentence
#     words = sentence.split()
#     stemmed_sentence = ' '.join(stemmer.stem(word) for word in words)
#     # Remove double spaces
#     #stemmed_sentence = re.sub(r'\s+', ' ', stemmed_sentence).strip()
#     stemmed_sentence = re.sub(r'\s+', ' ', stemmed_sentence).strip()
#     cleaned_sentences_stemmed.append(stemmed_sentence)


#creating list of unique words
unique_words = GetUniqueTokens(sentences_df['Stemmed Words'])

# if(remove_short_sentences):
#     sentences_df = sentences_df[sentences_df['Stemmed Words'].apply(lambda x: len(x) >= sentence_minimal_len)]

#creating term frequency matrix
#tf = np.zeros((len(cleaned_sentences), len(unique_words)), dtype=int)
# Creating an empty DataFrame with sentences as rows and unique words as columns
tf = pd.DataFrame(0, index=range(len(sentences_df)), columns=unique_words)

tf = CreateTermFrequencyMatrix(sentences_df,unique_words,tf)
#assigning values to each word
# word_index = 0
#
# for word in unique_words:
#     sentence_index = 0
#     for sentence in cleaned_sentences:
#         if word in sentence:
#             #print(sentence_index,word_index)
#             tf[sentence_index,word_index] += 1
#         else:
#             print(sentence, word)
#         sentence_index += 1
#     word_index += 1







#tf = tf.rename(columns=unique_words)

#checking if we don't have empty vectors
#non_empty_rows = ~np.all(tf == 0, axis=1)

# Filter the array to keep non-empty rows
#tf = tf[non_empty_rows]
#saving tf
#df = pd.DataFrame(tf)

# Specify the Excel file name and the sheet name
excel_file_name = "tf_v3.xlsx"
sheet_name = "Sheet1"

# Save the DataFrame to an Excel file
tf.to_excel(excel_file_name, sheet_name=sheet_name, index=False)

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



df = pd.DataFrame(similarity_matrix)

# Specify the Excel file name and the sheet name
excel_file_name = "cosine_similarity_2.xlsx"
#sheet_name = "Sheet1"

# Save the DataFrame to an Excel file
df.to_excel(excel_file_name, sheet_name=sheet_name, index=False)

#V = textrank(similarity_matrix, 400, 0.85)
nx_graph = nx.from_numpy_array(similarity_matrix)
scores = nx.pagerank(nx_graph, max_iter = 1000)
sentences_df["Ranks"] = scores
sorted_df = sentences_df.sort_values(by='Ranks', ascending=False)

# Select the top 20 sentences
top_20_sentences = sorted_df.head(20)

# Print the top 20 sentences
print("Top 20 Sentences with Highest Ranks:")
for index, row in top_20_sentences.iterrows():
    print(f"{index + 1}: {row['Original Sentence']} (Rank: {row['Ranks']})")

top_20_sentences.to_excel("Highest ranked sentences_2.xlsx", sheet_name = sheet_name, index = False)



