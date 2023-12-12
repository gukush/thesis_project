import fitz  # PyMuPDF
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import summary as sm
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import word_tokenize
from nltk.util import ngrams
from nltk.stem import PorterStemmer
import pandas as pd
#from streamlit import

try: 
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

stemmer = PorterStemmer()

#reading words from txt file
def read_ngrams_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    stemmed_ngrams = []
    for line in lines:
        line = line.strip()
        if line:
            # Tokenize the line into words
            words = word_tokenize(line)
            # Stem each word in the line
            stemmed_words = [stemmer.stem(word) for word in words]
            # Join the stemmed words back into an n-gram
            stemmed_ngram = ' '.join(stemmed_words)
            stemmed_ngrams.append(stemmed_ngram)

    return stemmed_ngrams

Financial_chapter_pattern = [r'Financial', r'Financial discussion', r'Item 7\.', ]
ESG_keywords = read_ngrams_from_file("/thesis_project/examples/ESG_word_list.txt")
#print(ESG_keywords)
Risk_keywords = ["Risk, risks", "threats"]
Risk_chapter_pattern = [r'Item 1A\.', r'Risk Factors', r'Risk']
#source https://dictionary.cambridge.org/thesaurus/goal
Goals_keywords = ["GOALS", "OBJECTIVES", "MISSION", "PLANNING", "PLANS FOR", "AIMS", "TARGETS", "THRESHOLDS", "HOPES", "ASPIRATIONS", "INTENTIONS", "NEXT YEAR", "PURPOSE", "INTENT"]

def sentence_contains_ngrams_stemmed(stemmed_words_list, ngrams_list):
    # Generate ngrams from the stemmed words list
    all_ngrams = []
    for n in range(1, 3):  # Adjust range as needed (1 for unigrams, 2 for bigrams, 3 for trigrams)
        all_ngrams.extend([' '.join(gram) for gram in ngrams(stemmed_words_list, n)])

    # Check if any of the ngrams are in the ngrams_list
    return any(gram in ngrams_list for gram in all_ngrams)

def extract_sentences_with_ngrams(ngrams_list, sentences_df):
    # Apply the function to each list of stemmed words in the DataFrame
    contains_ngrams = sentences_df['Stemmed Words'].apply(
        lambda stemmed_words: sentence_contains_ngrams_stemmed(stemmed_words, ngrams_list))

    # Filter the DataFrame for sentences that contain ngrams
    filtered_sentences_df = sentences_df[contains_ngrams]
    filtered_sentences_df = filtered_sentences_df.drop(
        columns=["Preprocessed Sentence", "Tokenized Sentence", "Stemmed Words"])
    return filtered_sentences_df

# Define a function that will check if any keyword is in the sentence
def sentence_contains_keywords(sentence, keywords):
    return any(keyword in sentence.lower() for keyword in keywords)

def find_page_numbers(toc_df, chapter_regex):
    # Filter rows where 'Chapter Name' contains '1A' or 'Risk Factors'
    def contains_pattern(chapter_name, chapter_regex):
        if not isinstance(chapter_name, str):
            return False
        return any(re.search(pattern, chapter_name, re.IGNORECASE) for pattern in chapter_regex)

    filtered = toc_df[toc_df['Chapter name'].apply(lambda x: contains_pattern(x, chapter_regex))]
    indices = filtered.index.tolist()
    page_numbers = {}

    for index in indices:
        # Get page number of the current row
        current_page = toc_df.loc[index, 'PDF index']
        next_page = toc_df.loc[index + 1, 'PDF index'] if index + 1 < len(toc_df) else None
        page_numbers[toc_df.loc[index, 'PDF index']] = (current_page, next_page)

    return page_numbers


def filter_sentences(sentences_df, chapter_df, chapter_regex):
    page_numbers = find_page_numbers(chapter_df, chapter_regex)

    # Check if page_numbers is empty
    if not page_numbers:
        # Handle the case (e.g., return an empty DataFrame or print a message)
        print("No matching chapters found.")
        return pd.DataFrame()

    # Proceed if page_numbers is not empty
    first_chapter_name = next(iter(page_numbers))
    start_page, end_page = page_numbers[first_chapter_name]

    if end_page:
        filtered_sentences = sentences_df[
            (sentences_df['PDF Page Number'] >= start_page) & (sentences_df['PDF Page Number'] <= end_page)]
    else:
        filtered_sentences = sentences_df[sentences_df['PDF Page Number'] == start_page]

    return filtered_sentences


def extract_sentences_with_keywords(keywords, sentences_df):
    # Ensure keywords are lowercased for matching
    keywords = [keyword.lower() for keyword in keywords]

    # Apply the function to each sentence in the DataFrame
    contains_keywords = sentences_df['Original Sentence'].apply(
        lambda sentence: sentence_contains_keywords(sentence, keywords))

    # Filter the DataFrame for sentences that contain keywords
    filtered_sentences_df = sentences_df[contains_keywords]
    filtered_sentences_df = filtered_sentences_df.drop(columns = ["Preprocessed Sentence", "Tokenized Sentence", "Stemmed Words"])
    #print(filtered_sentences_df)
    return filtered_sentences_df

def analyze_sentiment(sentence_df):
    # Initialize the SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()

    # Function to get sentiment score
    def get_sentiment_score(sentence_df):
        return sia.polarity_scores(sentence_df)['compound']

    # Apply the function to the specified column and create a new column for sentiment score
    sentence_df['Sentiment Score'] = sentence_df['Original Sentence'].apply(get_sentiment_score)

    return sentence_df

# Example usage:
# Assuming sentences_df is a DataFrame with a column named 'Sentences' containing the sentences.
# ESG_keywords are the keywords you're looking for in the sentences.

#filtered_sentences = extract_sentences_with_keywords(ESG_keywords, sentences_df)
