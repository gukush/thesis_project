import streamlit as st
import fitz
import table_extraction as tab
import thesis_summarizer as th
import extracting_data as ed
import css_like as front
import sentiment_analysis as sa
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

TEXTRANK_SUMMARY = 1
BART_SUMMARY = 2
ERROR_SUMMARY = 3

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
            if not st.session_state['whole_report']:
                bounded_report_text = th.importFileFromStream(report_data, st.session_state['num_start'],
                                                              st.session_state['num_end'])
                st.session_state['bounded_report_text'] = bounded_report_text
            if 'summary' not in st.session_state:
                #some default stuff
                pass
            elif st.session_state['summary'] == TEXTRANK_SUMMARY:
                # TODO - add logic for outputting summary into simple text
                #st.write("Here would be textrank summary")
                top20 = th.step_by_step(bounded_report_text)
                st.write(top20)
                st.write(st.session_state['report_page_count'])
            elif st.session_state['summary'] == BART_SUMMARY:
                # TODO - make the import process make more sense (it loads for a long time initially)
                import bart_summarizer
                if 'num_sentences' not in st.session_state:
                    num_sentences = 20
                else:
                    num_sentences = st.session_state['num_sentences']
                report_summary = bart_summarizer.abstractive(bounded_report_text,num_sentences)
                st.write(report_summary)

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
        elif 'BART' in option:
            st.session_state['summary'] = BART_SUMMARY
        else:
            st_session_state['summary'] = ERROR_SUMMARY
        #extractive_summary = st.checkbox("Extractive Summary")
        st.session_state['basic_analysis'] = st.checkbox("Basic Analysis", True)
        advanced_analysis = st.checkbox("Advanced Analysis")
        st.session_state['num_sentences'] = st.number_input("Number of sentences in summary", min_value=1, value=1, step=1)

    with col2:
        analyst = st.checkbox("Analyst")
        investor = st.checkbox("Investor")
        shareholder = st.checkbox("Shareholder")

    with col3:
        st.session_state['whole_report'] = st.checkbox("Whole Report", False)
        st.session_state['num_start'] = st.number_input("Start from page (0-indexed)", min_value=0, value=0, step=1, key='start_page')
        st.session_state['num_end'] = st.number_input("End at page (0-indexed)", min_value=0, value=5, step=1, key='end_page')

    # Processing the uploaded file (placeholder logic)
    if uploaded_file is not None:
        #st.write("File processing logic goes here.")
        st.session_state['uploaded_file'] = uploaded_file


def show_basic_analysis():
    st.title('Basic Analysis')

    if all((key in st.session_state) and (key is not None) for key in ('basic_analysis','uploaded_file','report_text')):# st.session_state['basic_analysis'] and st.session_state['uploaded_file']: // changed it because it crashes when file was not uploaded
        st.session_state['company_name'] = ed.get_company_name(st.session_state['report_text'])
        #print(st.session_state['report_text'])
        st.write(st.session_state['company_name'])
        col1, col2 = st.columns(2)

        with col1:
            st.subheader('Company Details')
            company = st.text_input('Company', st.session_state['company_name'])
            year = st.number_input('Year', value=0000)
            report_type = st.selectbox('Type', ['ANNUAL REPORT', 'FINANCIAL STATEMENT', 'OTHER'])
            industry = st.selectbox('Industry', ['INSURANCE', 'TECHNOLOGY', 'HEALTHCARE'])
            pages = st.number_input('Pages', value=st.session_state['report_page_count'])
            avg_words = st.number_input('Average Words per Page', value= st.session_state['Word_count']/st.session_state['report_page_count'])
            reading_time = st.number_input('Reading Time', value = st.session_state['Word_count']/250)
            sentences = st.number_input('Number of Sentences', value=len(st.session_state['preprocessed_df']))

        # You can add functionality to process and update these details as needed.
        with col2:
            st.subheader('Key Word Analysis of the Report')

            # Assuming you have the keywords and their frequencies
            #keywords = ['fun', 'easy', 'inclusive', 'share', 'software', 'live', 'beautiful', 'reflection', 'thoughts',
            #            'interactive', 'brainstorm', 'knowledge', 'ideas', 'ice breaker']
           # frequencies = [5, 4, 3, 3, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1]

            tf_wordloud = st.session_state['tf_wordcloud']
            st.write(tf_wordloud)
            wordcloud = WordCloud(width=800, height=400).generate_from_frequencies(tf_wordloud)
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            st.pyplot(plt)

            # For the bar chart, assuming you have the data
            data = pd.Series([0.2, 0.15, 0.10, 0.05, 0.03, 0.02, 0.01, 0.005, 0.003],
                             index=[1, 2, 3, 4, 5, 6, 7, 8, 9])
            st.bar_chart(data)
        # TODO: is this in correct section? I moved it
        additional_topic = st.text_area('Add Additional Topic for Modelling (Optional)')
        if st.button('Process Report'):
            # Here you can define what processing occurs when the button is clicked
            st.write(f'Processing report for {company}...')
            # Add your processing functions here
    else:
        st.write("Please upload file in \"Additional selection\" page and then run the pipeline!")
 


def show_advanced_analysis():
    # Advanced Analysis Tab Content
    st.header("Advanced Analysis")
    if 'report_text' in st.session_state:
        ESG_sentences = sa.extract_sentences_with_keywords(sa.ESG_keywords, st.session_state['preprocessed_df'])
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
            #st.write(ESG_sentences)
            ESG_sentiment_df = sa.analyze_sentiment(ESG_sentences)
            st.write(ESG_sentiment_df)
            st.write("Average sentiment score", ESG_sentiment_df.sum("Sentiment Score"))
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
                model_structure, model_detection, image_processor = tab.initializeTable()
                csv_strings = tab.TableExtractionFromStream(stream=report_data, keywords=tab.keywords,
                                                            num_start=st.session_state['num_start'],num_end=st.session_state['num_end'],
                                                            model_structure = model_structure, model_detection = model_detection,
                                                            image_processor = image_processor
                                                            ) 
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
    show_basic_analysis()
elif app_mode == "Advanced Analysis":
    show_advanced_analysis()

