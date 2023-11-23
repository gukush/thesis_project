import streamlit as st
import re


def find_years(text):
    # Regular expression pattern for a year
    year_pattern = r'\b(?:[1-2][0-9]{3})\b'

    # Find all occurrences of the pattern in the text
    years = re.findall(year_pattern, text)

    return years

    # Join the words using the specified operator
    return operator.join(words)

def getPageNum(file_name):
    #TODO: add support for extracting range of pages (currenlty extracts whole doc)
    with fitz.open(report_path) as pdf:
        # Extract text from the first page
            page_count = doc.page_count
        return page_count

report_data = st.session_state['uploaded_file'].getvalue()
name_of_the_report = st.session_state['uploaded_file'].name
number_of_pages = getPageNum(name_of_the_report)
text = st.session_state['report_text']
words_count = lengt