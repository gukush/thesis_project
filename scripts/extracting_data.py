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
    return operator.join(words)

def get_company_name(text):
    extract_companies_name = r"\b[A-Z]\w+(?:\.com?)?(?:[ -]+(?:&[ -]+)?[A-Z]\w+(?:\.com?)?){0,2}[,\s]+(?i:ltd|llc|inc|plc|co(?:rp)?|group|holding|gmbh)\b"
    matches = re.findall(extract_companies_name, text)
    #we count occurences of companies mentioned
    count_dict = Counter(matches)
    company_name = max(count_dict, key = count_dict.get)
    return company_name

#st.session_state['company_name'] = get_company_name(st.session_state['bounded_report_text'])
# report_data = st.session_state['uploaded_file'].getvalue()
# name_of_the_report = st.session_state['uploaded_file'].name
# number_of_pages = getPageNum(name_of_the_report)
# text = st.session_state['report_text']
# words_count = lengt