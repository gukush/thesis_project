import streamlit as st
import re
from collections import Counter
import streamlit as st

st.session_state['company_name'] = 'NA'

def find_years(text):
    # Regular expression pattern for a year
    year_pattern = r'\b(?:[1-2][0-9]{3})\b'

    # Find all occurrences of the pattern in the text
    years = re.findall(year_pattern, text)
    count_dict = Counter(years)
    year_of_report = max(count_dict, key=count_dict.get)
    return year_of_report

def get_company_name(text):
    extract_companies_name = r"\b[A-Z]\w+(?:\.com?)?(?:[ -]+(?:&[ -]+)?[A-Z]\w+(?:\.com?)?){0,2}[,\s]+(?i:ltd|llc|inc|plc|co(?:rp)?|group|holding|gmbh)\b"
    matches = re.findall(extract_companies_name, text)
    #We count occurences of companies mentioned
    count_dict = Counter(matches)
    company_name = max(count_dict, key = count_dict.get)
    #there is a common problem that the group is identified as a company name, in such case we take second most occuring word
    if company_name.lower() != "the group" :
        return company_name
    else:
        # Find the second most common company name
        if len(count_dict) > 1:
            second_most_common_name, _ = count_dict.most_common(2)[1]
            return second_most_common_name
        else:
            return None

def count_report_bigrams(text):
    # Define bigrams for each report type
    bigrams_10k = [('form', '10'), ('form', '10-k'), ('form', '10 - k'), ('risk', 'factors'), ('10-k', '')]
    bigrams_annual_report = [('annual', 'report'), ('financial', 'statement')]
    bigrams_sustainability = [('CSR', ' '), ('corporate', 'social'), ('social', 'development'), ('sustainability', 'report'), ('social','responsibility')]
    # Example bigrams for other reports

    # Function to find bigrams in the text
    def find_bigrams(text, bigram_list):
        # Clean and tokenize the text
        words = re.findall(r'\b\w+\b', text.lower())
        # Create bigrams from the words
        bigrams = zip(words, words[1:])
        bigrams_count = Counter(bigrams)
        return sum(bigrams_count[bg] for bg in bigram_list)

    # Count bigrams for each report type
    counts_10k = find_bigrams(text, bigrams_10k)
    counts_annual_report = find_bigrams(text, bigrams_annual_report)
    counts_sustainability = find_bigrams(text, bigrams_sustainability)
    #even if the report is not of a given type it might consist of keywords, that's why we add the threshold
    count_threshold = 5

    if counts_10k > 0:
        return "10-K Report"
    else:
        if counts_annual_report > counts_sustainability:
            if counts_annual_report > 5:
                return "Annual Report"
        else:
            if counts_sustainability > 5:
                return "Sustainability Report"
        return "Other Report"

def create_barchart():
    if 'summary_results' in st.session_state and st.session_state['summary'] == 1:
        ranks_df = st.session_state['summary_results'].copy()
        ranks_sum = ranks_df.groupby('PDF Page Number')['Ranks'].mean()
        sorted_ranks_sum = ranks_sum.sort_values(ascending=False)
        sorted_ranks_dict = sorted_ranks_sum.to_dict()

    return sorted_ranks_dict