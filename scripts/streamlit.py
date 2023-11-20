import streamlit as st
import fitz 
import thesis_summarizer as th

TEXTRANK_SUMMARY = 1
BART_SUMMARY = 2
# Function for the main content
def show_main_content():
    st.write("$\\frac{a}{b}$")
    st.header("Welcome in pipeline for extracting and summarizing data from business reports!")
    if st.button('Run pipeline'):
        st.write('Pipeline is being run')
        if 'uploaded_file' not in st.session_state:
            st.write('Please upload report')
        else:
            report_data = st.session_state['uploaded_file'].getvalue()
            report_text = th.importFileFromStream(report_data) 
            if 'summary' not in st.session_state:
                #some default stuff
                pass
            elif st.session_state['summary'] == TEXTRANK_SUMMARY:
                # TODO - add logic for outputting summary into simple text
                st.write("Here would be textrank summary")
            else:
                # TODO - make the import process make more sense (it loads for a long time initially)
                import bart_summarizer
                report_summary = bart_summarizer.abstractive(report_text)
                st.write(report_summary[0]['summary_text'])
            #st.write(report_text)
            
    # Your main page content here
# TODO - move the st.write calls that show summary results to this section (too tired rn)
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
        option = st.selectbox('Choose type of summarization',('Extractive (textrank)','Abstractive (BART)'),index=0)
        if 'textrank' in option:
            st.session_state['summary'] = TEXTRANK_SUMMARY
        else:
            st.session_state['summary'] = BART_SUMMARY
        #extractive_summary = st.checkbox("Extractive Summary")
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
        st.session_state['uploaded_file'] = uploaded_file


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
