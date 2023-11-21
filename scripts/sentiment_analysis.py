import fitz  # PyMuPDF
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import summary as sm
#from streamlit import


# Initialize stemmer
stemmer = PorterStemmer()
def preprocessing(text,sentence_minimal_len = 5):
    # Import the file and extract text
    #text = sm.importFile(page_num_down, page_num_up, file_name)

    # Split the text into sentences
    sentences = sm.splitTextIntoSentences(text)

    # Assuming sentences is a list of strings, convert it to a DataFrame
    import pandas as pd
    sentences_df = pd.DataFrame(sentences, columns=['Original Sentences'])

    # Clean each sentence
    sentences_df['Cleaned Sentences'] = sentences_df['Original Sentences'].apply(sm.CleanText)

    # Tokenize sentences
    sentences_df['Tokenized Sentence'] = sentences_df['Cleaned Sentences'].apply(lambda x: x.split())

    # Remove stopwords
    sentences_df = sm.RemoveStopWords(sentences_df)

    # Remove short sentences
    sentences_df['Preprocessed Sentences'] = sm.RemoveShortSentences(sentences_df, sentence_minimal_len)

    # Stem the words
    sentences_df['Stemmed Words'] = sentences_df['Tokenized Sentence'].apply(sm.StemWords)

    # Get unique tokens
    unique_tokens = sm.GetUniqueTokens(sentences_df['Stemmed Words'])

    # Return the preprocessed DataFrame and the unique tokens
    print("preprocessing finished")
    return sentences_df, unique_tokens

#sentences_df = preprocessing()
#TODO stemming of keywords
def extract_sentences_with_keywords(keywords, sentences_df):
    # Ensure keywords are lowercased for matching
    keywords = [keyword.lower() for keyword in keywords]

    # Define a function that will check if any keyword is in the sentence
    def sentence_contains_keywords(sentence):
        return any(keyword in sentence.lower() for keyword in keywords)

    # Apply the function to each sentence in the DataFrame
    contains_keywords = sentences_df['Sentences'].apply(sentence_contains_keywords)

    # Filter the DataFrame for sentences that contain keywords
    filtered_sentences_df = sentences_df[contains_keywords]
    print(filtered_sentences_df)
    return filtered_sentences_df


# Example usage:
# Assuming sentences_df is a DataFrame with a column named 'Sentences' containing the sentences.
# ESG_keywords are the keywords you're looking for in the sentences.
ESG_keywords = ["social", "global", "environmental", "ESG", "esg"]
#filtered_sentences = extract_sentences_with_keywords(ESG_keywords, sentences_df)