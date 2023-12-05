import fitz  # PyMuPDF
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import summary as sm
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
#from streamlit import

try: 
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')


ESG_keywords = ["social", "global", "environmental", "ESG", "esg"]
Risk_keywords = ["Risk, risks", "threats"]
Goals_keywords = ["Goals", "objectives", "mission"]

# Define a function that will check if any keyword is in the sentence
def sentence_contains_keywords(sentence, keywords):
    return any(keyword in sentence.lower() for keyword in keywords)


#sentences_df = preprocessing()
#TODO stemming of keywords
#def extract_given_chapter(keyword_1, keyword_2):


def extract_sentences_with_keywords(keywords, sentences_df):
    # Ensure keywords are lowercased for matching
    keywords = [keyword.lower() for keyword in keywords]

    # Apply the function to each sentence in the DataFrame
    contains_keywords = sentences_df['Original Sentence'].apply(
        lambda sentence: sentence_contains_keywords(sentence, keywords))

    # Filter the DataFrame for sentences that contain keywords
    filtered_sentences_df = sentences_df[contains_keywords]
    filtered_sentences_df = filtered_sentences_df.drop(columns = ["Preprocessed Sentence", "Tokenized Sentence", "Stemmed Words", "Ranks"])
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
