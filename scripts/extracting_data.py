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

    # Join the words using the specified operator
    #return operator.join(words)

def get_company_name(text):
    extract_companies_name = r"\b[A-Z]\w+(?:\.com?)?(?:[ -]+(?:&[ -]+)?[A-Z]\w+(?:\.com?)?){0,2}[,\s]+(?i:ltd|llc|inc|plc|co(?:rp)?|group|holding|gmbh)\b"
    matches = re.findall(extract_companies_name, text)
    #we count occurences of companies mentioned
    count_dict = Counter(matches)
    company_name = max(count_dict, key = count_dict.get)
    return company_name

def create_barchart():
    if 'summary_results' in st.session_state and st.session_state['summary'] == 1:
        ranks_df = st.session_state['summary_results'].copy()
        ranks_sum = ranks_df.groupby('PDF Page Number')['Ranks'].mean()
        sorted_ranks_sum = ranks_sum.sort_values(ascending=False)
        sorted_ranks_dict = sorted_ranks_sum.to_dict()

    return sorted_ranks_dict