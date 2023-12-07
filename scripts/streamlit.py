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
import io

TEXTRANK_SUMMARY = 1
BART_SUMMARY = 2
ERROR_SUMMARY = 3

front.generate_custom_styles()

# def process_report(uploaded_file, summary_type, num_start, num_end, num_sentences):
#     if uploaded_file is None:
#         return "Please upload a report"
#     else:
#         report_data = uploaded_file.getvalue()
#         st.session_state["report_text"] = th.importFileFromStream(report_data)
#         bounded_report_text = th.importFileFromStream(report_data, num_start, num_end)
#         st.session_state["bounded_report_text"] = bounded_report_text
#
#         if summary_type == TEXTRANK_SUMMARY:
#             top20 = th.step_by_step(bounded_report_text)
#             return top20
#         elif summary_type == BART_SUMMARY:
#             import bart_summarizer
#             report_summary = bart_summarizer.abstractive(bounded_report_text, num_sentences)
#             return report_summary
#         else:
#             return "Invalid summary type"

def process_report(uploaded_file, summary_type, num_start, num_end, num_sentences):
    if uploaded_file is None:
        return "Please upload a report"
    else:
        report_data = uploaded_file
        #st.session_state["report_text"] = th.importFileFromStream(report_data)
        bounded_report_text = th.importFileFromStream(report_data, num_start, num_end)
        st.session_state["bounded_report_text"] = bounded_report_text

        if summary_type == TEXTRANK_SUMMARY:
            top20 = th.step_by_step(bounded_report_text)
            return top20
        elif summary_type == BART_SUMMARY:
            import bart_summarizer
            report_summary = bart_summarizer.abstractive(bounded_report_text, num_sentences)
            return report_summary
        else:
            return "Invalid summary type"


# Function for the main content
def show_main_content():
    st.header("Welcome in pipeline for extracting and summarizing data from business reports!")
    st.header("Please use the navigation panel to move through sections")
    st.header("To run the pipeline go to advanced selection panel, and configure your output")

# Function for the additional selection page
def show_additional_selection():
    st.header("Additional Selection")

    # File uploader
    #uploaded_file = st.file_uploader("Test upload")
    uploaded_file = st.text_input("Test upload")

    # Input fields and variable/s
    col1, col2, col3 = st.columns(3)

    with col1:
        option = st.selectbox('Choose type of summarization',('Extractive (textrank)','Abstractive (BART)'),index=0)
        if 'textrank' in option:
            st.session_state['summary'] = TEXTRANK_SUMMARY
        elif 'BART' in option:
            st.session_state['summary'] = BART_SUMMARY
        else:
            st.session_state['summary'] = ERROR_SUMMARY
        #extractive_summary = st.checkbox("Extractive Summary")
        st.session_state['basic_analysis'] = st.checkbox("Basic Analysis", True)
        advanced_analysis = st.checkbox("Advanced Analysis")
        st.session_state['num_sentences'] = st.number_input("Number of sentences in summary", min_value=1, value=6, step=1)

        # Custom stopwords input
        custom_stopwords = st.text_area("Enter custom stopwords (separated by commas)")
        if custom_stopwords:
            st.session_state['custom_stopwords'] = [word.strip() for word in custom_stopwords.split(',')]
            # Button to clear stopwords
        if st.button("Clear stopwords") and 'custom_stopwords' in st.session_state:
            st.session_state['custom_stopwords'] = []
            st.write("Custom stopwords have been cleared.")
    # with col2:
    #     analyst = st.checkbox("Analyst")
    #     investor = st.checkbox("Investor")
    #     shareholder = st.checkbox("Stakeholder")

       #st.write("Customize your output")

    with col3:
        st.session_state['whole_report'] = st.checkbox("Whole Report", False)
        st.session_state['num_start'] = st.number_input("Start from page (0-indexed)", min_value=0, value=0, step=1, key='start_page')
        st.session_state['num_end'] = st.number_input("End at page (0-indexed)", min_value=0, value=5, step=1, key='end_page')
        # Custom keywords input
        st.write("\n\n")
        st.write("\n\n")
        #st.write("\n\n")
        custom_keywords = st.text_area("Enter custom keywords (separated by commas)")
        if custom_keywords:
            st.session_state['custom_keywords'] = [word.strip() for word in custom_keywords.split(',')]
            st.write(custom_keywords)
        if st.button("Clear keywords") and 'custom_keywords' in st.session_state:
            st.session_state['custom_keywords'] = []
            st.write("Custom keywords have been cleared.")
    # Processing the uploaded file (placeholder logic)
    if st.button('Run pipeline') and uploaded_file is not None:
        #st.write("File processing logic goes here.")
        st.session_state['uploaded_file'] = uploaded_file
        st.session_state['summary_results'] = process_report(uploaded_file, st.session_state['summary'], st.session_state['num_start'], st.session_state['num_end'], st.session_state['num_sentences'] )
        st.write("Your summary is ready. Visit summary tab to see it!")
    elif uploaded_file is None:
        st.write("Please add you file")


def show_basic_analysis():
    st.title('Basic Analysis')

    if all((key in st.session_state) and (key is not None) for key in ('basic_analysis','uploaded_file','report_text')):# st.session_state['basic_analysis'] and st.session_state['uploaded_file']: // changed it because it crashes when file was not uploaded
        st.session_state['company_name'] = ed.get_company_name(st.session_state['report_text'])
        st.session_state['report_year'] = ed.find_years(st.session_state['report_text'])
        #print(st.session_state['report_text'])
        st.write(st.session_state['company_name'])
        col1, col2 = st.columns(2)
        page_num_selected = st.session_state['num_end'] - st.session_state['num_start']
        with col1:
            st.subheader('Company Details')
            company = st.text_input('Company', st.session_state['company_name'])
            year = st.text_input('Year', st.session_state['report_year'])
            report_type = st.selectbox('Type', ['Annual report', '10-K report', 'OTHER'])
            #industry = st.selectbox('Industry', ['INSURANCE', 'TECHNOLOGY', 'HEALTHCARE'])
            pages_whole = st.number_input('Pages (whole report)', value=st.session_state['report_page_count'])
            if(st.session_state['whole_report'] == False):
                pages_selected = st.number_input('Pages (selected part)', value=page_num_selected)
            avg_words = st.number_input('Average Words per Page (selected part)', value= st.session_state['Word_count']/page_num_selected)
            sentences = st.number_input('Number of Sentences (in the selected part)', value=len(st.session_state['preprocessed_df']))
            reading_time = st.number_input('Reading Time (in minutes)', value = st.session_state['Word_count']/250)


        # You can add functionality to process and update these details as needed.
        with col2:
            st.subheader('Key Word Analysis of the Report')
            tf_wordcloud = st.session_state['tf_wordcloud']
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(tf_wordcloud)
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            st.pyplot(plt)

            #Barchart with most important pages from summary
            st.subheader("Most important PDF pages in a given page set")
            data = ed.create_barchart()
            st.bar_chart(data)
    else:
        st.write("Please upload file in \"Additional selection\" page and then run the pipeline!")

# Function for the summary results
def show_summary_results():
    st.header("Summary Results")
    # Your summary results content here
    if "summary_results" in st.session_state and st.session_state['summary'] == TEXTRANK_SUMMARY :
        st.write(st.session_state["summary_results"].head(st.session_state['num_sentences']))
        st.write(th.print_Text_Rank_as_text(st.session_state["summary_results"].head(st.session_state['num_sentences'])))
    elif "summary_results" in st.session_state and st.session_state['summary'] == BART_SUMMARY:
        st.write(st.session_state["summary_results"])
    else:
        st.write("Summary results not available.")
        #st.write(st.session_state["similarity_matrix"])

def show_advanced_analysis():
    # Advanced Analysis Tab Content
    st.header("Advanced Analysis")

    if 'bounded_report_text' in st.session_state:
        ESG_sentences = sa.extract_sentences_with_keywords(sa.ESG_keywords, st.session_state['preprocessed_df'])
        Goals_sentences = sa.extract_sentences_with_keywords(sa.Risk_keywords, st.session_state['preprocessed_df'])
        Risks_sentences = sa.extract_sentences_with_keywords(sa.Goals_keywords, st.session_state['preprocessed_df'])

        col1, col2 = st.columns(2)
        ESG_sentiment_df = sa.analyze_sentiment(ESG_sentences)
        st.write(ESG_sentiment_df)
        average_sentiment = ESG_sentiment_df["Sentiment Score"].mean()
        st.write("Average sentiment score", average_sentiment)
        st.session_state['average_sentiment'] = average_sentiment
        # Slider
        sentiment_value = st.slider("ESG Sentiment Score", min_value=-1.0, max_value=1.0,
                                    value=float(average_sentiment), disabled=True)
        color = front.get_slider_color(sentiment_value)
        color_html = f"<div style='width: 50px; height: 20px; background-color: {color};'></div>"
        st.markdown(color_html, unsafe_allow_html=True)
        # Display the color indicator

    #st.markdown('</div>', unsafe_allow_html=True)

    # Goals and Objectives Section
    with col2:
        st.markdown('<div class="goals-section">', unsafe_allow_html=True)
        st.subheader('Goals and Objectives')
        st.text_area('Main Goals, based on the ranks from Text Rank')
        st.markdown('</div>', unsafe_allow_html=True)

        Goals_sentiment_df = sa.analyze_sentiment(Goals_sentences)
        st.write(Goals_sentiment_df)
        average_goals_sentiment = Goals_sentiment_df["Sentiment Score"].mean()
        st.write("Average sentiment score", average_goals_sentiment)
        # Slider
        sentiment_value = st.slider("Goals and objectives Sentiment Score", min_value=-1.0, max_value=1.0,
                                    value=float(average_goals_sentiment), disabled=True)
        color = front.get_slider_color(sentiment_value)
        color_html = f"<div style='width: 50px; height: 20px; background-color: {color};'></div>"
        st.markdown(color_html, unsafe_allow_html=True)

        # Risks and Threats Section
        st.markdown('<div class="risks-section">', unsafe_allow_html=True)
        st.subheader('Risks and Threats')
        st.text_area('Sentiment of sentences about Risks and Threats')
        st.markdown('</div>', unsafe_allow_html=True)

        Risks_sentiment_df = sa.analyze_sentiment(Risks_sentences)
        st.write(Risks_sentiment_df)
        average_risks_sentiment = Risks_sentiment_df["Sentiment Score"].mean()
        st.write("Average sentiment score", average_risks_sentiment)
        # Slider
        sentiment_value = st.slider("ESG Sentiment Score", min_value=-1.0, max_value=1.0,
                                    value=float(average_risks_sentiment), disabled=True)
        color = front.get_slider_color(sentiment_value)
        color_html = f"<div style='width: 50px; height: 20px; background-color: {color};'></div>"
        st.markdown(color_html, unsafe_allow_html=True)

def show_table_extraction():
    # Extract Table Button at the bottom
    if st.button("Extract Table (If it works)"):
        if 'uploaded_file' not in st.session_state:
            st.write('Please upload report')
        else:
            report_data = st.session_state['uploaded_file'].getvalue()
            # TODO: make initialization once per session
            model_structure, model_detection, image_processor = tab.initializeTable()
            st.session_state['csv_strings'] = tab.TableExtractionFromStream(stream=report_data, keywords=tab.keywords,
                                                        num_start=st.session_state['num_start'],num_end=st.session_state['num_end'],
                                                        model_structure = model_structure, model_detection = model_detection,
                                                        image_processor = image_processor
                                                        )
            n_table = 0
            for i, csv_string in st.session_state['csv_strings']:
                n_table = n_table + 1
                st.write(f"Found on page {i}")
                st.write(pd.read_csv(io.StringIO(csv_string)))
                st.download_button(
                        "Download csv file",
                        csv_string,
                        f"table_{n_table}.csv",
                        "text/csv"
                        )
            st.write("done")


# Sidebar for navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Choose the Page",
                            ["Main Page", "Summary Results", "Additional Selection", "Basic analysis", "Advanced Analysis", "Table Extraction" ])

# Page routing based on sidebar selection
if app_mode == "Main Page":
    show_main_content()
elif app_mode == "Additional Selection":
    show_additional_selection()
elif app_mode == "Summary Results":
    show_summary_results()
elif app_mode == "Basic analysis":
    show_basic_analysis()
elif app_mode == "Advanced Analysis":
    show_advanced_analysis()
elif app_mode == "Table Extraction":
    show_table_extraction()

