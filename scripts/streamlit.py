import streamlit as st
import fitz
import table_extraction as tab
import thesis_summarizer as th
import css_like as front
import sentiment_analysis as sa
import pandas as pd

TEXTRANK_SUMMARY = 1
BART_SUMMARY = 2

front.generate_custom_styles()

# set initial session state (TODO: might put into separate function)
if 'num_start' not in st.session_state:
    st.session_state['num_start'] = None
if 'num_end' not in st.session_state:
    st.session_state['num_end'] = None

# Function for the main content
def show_main_content():
    st.header("Welcome in pipeline for extracting and summarizing data from business reports!")
    if st.button('Run pipeline'):
        st.write('Pipeline is being run')
        if 'uploaded_file' not in st.session_state:
            st.write('Please upload report')
        else:
            report_data = st.session_state['uploaded_file'].getvalue()
            report_text = th.importFileFromStream(report_data)
            st.session_state['report_text'] = report_text
            if 'summary' not in st.session_state:
                #some default stuff
                pass
            elif st.session_state['summary'] == TEXTRANK_SUMMARY:
                # TODO - add logic for outputting summary into simple text
                #st.write("Here would be textrank summary")
                top20 = th.step_by_step(report_text)
                st.write(top20)
            else:
                # TODO - make the import process make more sense (it loads for a long time initially)
                import bart_summarizer
                report_summary = bart_summarizer.abstractive(report_text,1000)
                st.write(report_summary)
                #st.write(report_summary[0]['summary_text'])
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

    # Input fields and variable/s
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
        st.session_state['num_start'] = st.number_input("Start from page (0-indexed)", min_value=0, value=0, step=1, key='start_page')
        st.session_state['num_end'] = st.number_input("End at page (0-indexed)", min_value=0, value=0, step=1, key='end_page')

    # Processing the uploaded file (placeholder logic)
    if uploaded_file is not None:
        st.write("File processing logic goes here.")
        st.session_state['uploaded_file'] = uploaded_file

def show_advanced_analysis():
    # Advanced Analysis Tab Content
    st.header("Advanced Analysis")
    if 'report_text' in st.session_state:
        text = st.session_state['report_text']
        sentences_df = sa.preprocessing(text, 5)
        #sentences_df = sa.extract_sentences_with_keywords(sa.ESG_keywords, sentences_df)
    col1, col2 = st.columns(2)

    # Financial Section
    with col1:
        st.markdown('<div class="financial-section">', unsafe_allow_html=True)
        st.subheader('Financial')
        st.text_input('Total Revenue')
        st.text_input('Total Assets')
        st.markdown('</div>', unsafe_allow_html=True)

        # ESG Section
        st.markdown('<div class="esg-section">', unsafe_allow_html=True)
        st.subheader('ESG')
        st.text_area('Sentiment of sentences about Environment')
        #TODO wymyślić sprytniejszy warunek
        if 'report_text' in st.session_state:
            st.dataframe(sentences_df)
        st.markdown('</div>', unsafe_allow_html=True)

        # Goals and Objectives Section
    with col2:
        st.markdown('<div class="goals-section">', unsafe_allow_html=True)
        st.subheader('Goals and Objectives')
        st.text_area('Main Goals, based on the ranks from Text Rank')
        st.markdown('</div>', unsafe_allow_html=True)

        # Risks and Threats Section
        st.markdown('<div class="risks-section">', unsafe_allow_html=True)
        st.subheader('Risks and Threats')
        st.text_area('Sentiment of sentences about Risks and Threats')
        st.markdown('</div>', unsafe_allow_html=True)

    # Extract Table Button at the bottom
        if st.button("Extract Table (If it works)"):
            if 'uploaded_file' not in st.session_state:
                st.write('Please upload report')
            else:
                report_data = st.session_state['uploaded_file'].getvalue()
                csv_strings = tab.TableExtractionFromStream(stream=report_data, keywords=tab.keywords,num_start=st.session_state['num_start'],num_end=st.session_state['num_end']) 
                for csv_string in csv_strings:
                    st.write(csv_string)
                st.write("done")


# Sidebar for navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Choose the Page",
                            ["Main Page", "Summary Results", "Additional Selection", "Basic analysis", "Advanced Analysis" ])

# Page routing based on sidebar selection
if app_mode == "Main Page":
    show_main_content()
elif app_mode == "Summary Results":
    show_summary_results()
elif app_mode == "Additional Selection":
    show_additional_selection()
elif app_mode == "Basic analysis":
    show_additional_selection()
elif app_mode == "Advanced Analysis":
    show_advanced_analysis()

