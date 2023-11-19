import streamlit as st
import fitz

# Function for the main content
def show_main_content():
    st.header("Welcome in pipeline for extracting and summarizing data from business raport!")
    # Your main page content here

# Function for the summary results
def show_summary_results():
    st.header("Summary Results")
    # Your summary results content here

# Function for the additional selection page
def show_additional_selection():
    st.header("Additional Selection")

    # File uploader
    uploaded_file = st.file_uploader("Test upload")

    # Input fields and variables
    col1, col2, col3 = st.columns(3)

    with col1:
        extractive_summary = st.checkbox("Extractive Summary")
        basic_analysis = st.checkbox("Basic Analysis")
        advanced_analysis = st.checkbox("Advanced Analysis")
        num_sentences = st.number_input("Number of sentences in summary", min_value=1, value=1, step=1)

    with col2:
        analyst = st.checkbox("Analyst")
        investor = st.checkbox("Investor")
        shareholder = st.checkbox("Shareholder")

    with col3:
        whole_report = st.checkbox("Whole Report")
        start_page = st.number_input("Start from page (0-indexed)", min_value=0, value=0, step=1, key='start_page')
        end_page = st.number_input("End at page (0-indexed)", min_value=0, value=0, step=1, key='end_page')

    # Processing the uploaded file (placeholder logic)
    if uploaded_file is not None:
        st.write("File processing logic goes here.")

# Sidebar for navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Choose the Page",
                            ["Main Page", "Summary Results", "Additional Selection"])

# Page routing based on sidebar selection
if app_mode == "Main Page":
    show_main_content()
elif app_mode == "Summary Results":
    show_summary_results()
elif app_mode == "Additional Selection":
    show_additional_selection()
