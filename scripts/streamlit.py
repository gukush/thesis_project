import streamlit as st
import fitz
import table_extraction as tab
import thesis_summarizer as th
import extracting_data as ed
import css_like as front
import sentiment_analysis as sa
import test_toc as toc
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import io

TEXTRANK_SUMMARY = 1
BART_SUMMARY = 2
ERROR_SUMMARY = 3

front.generate_custom_styles()

def process_report(uploaded_file, summary_type, num_start, num_end, num_sentences):
    if uploaded_file is None:
        return "Please upload a report"
    else:
        report_data = uploaded_file.getvalue()
        st.session_state["report_text"] = th.importFileFromStream(report_data)
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

# def process_report(uploaded_file, summary_type, num_start, num_end, num_sentences):
#     if uploaded_file is None:
#         return "Please upload a report"
#     else:
#         report_data = uploaded_file
#         #st.session_state["report_text"] = th.importFileFromStream(report_data)
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


# Function for the main content
def show_main_content():
    st.header("Welcome in pipeline for extracting and summarizing data from business reports!")
    st.header("Please use the navigation panel to move through sections")
    st.header("To run the pipeline go to advanced selection panel, and configure your output")


# def update_selected_pages(index):
#     st.session_state['num_start'] = st.session_state["toc"].iloc[index, "PDF index"]
#     end_page= st.session_state["toc"].iloc[index + 1, "PDF index"] if index + 1 < len(st.session_state["toc"]) else None
#     st.session_state['num_end'] = end_page - 1

def update_selected_pages(index):
    st.session_state['num_start'] = st.session_state["toc"].iloc[index]['PDF index']
    st.session_state['num_end'] = st.session_state["toc"].iloc[index + 1]['PDF index'] if index + 1 < len(st.session_state["toc"]) else None
    if st.session_state['num_start'] == st.session_state['num_end']:
        st.session_state['num_end'] += 1

def flatten_chapter_names(df):
    def join_list_elements(lst):
        # Check if the element is a list
        if isinstance(lst, list):
            # Join the list elements into a string
            return ' '.join(map(str, lst))
        return lst  # Return as is if not a list

    # Apply this function to the 'Chapter name' column
    df['Chapter name'] = df['Chapter name'].apply(join_list_elements)
    return df

# Function for the additional selection page
def show_additional_selection():
    st.header("Additional Selection")

    # File uploader
    uploaded_file = st.file_uploader("Upload your report")

    #if "toc" in st.session_state:
        #st.write(st.session_state["toc"])
    if uploaded_file is not None and st.session_state['whole_report'] == False:
        #st.write("plik jest")
        doc = fitz.Document(stream=uploaded_file.getvalue())
        best_candidate = toc.find_best_candidate(doc)
        #We add toc to global variables and convert names in form of list to string
        st.session_state["toc"] = toc.get_toc_df(best_candidate)
        st.session_state["toc"] = flatten_chapter_names(st.session_state["toc"])
        print(st.session_state["toc"])
        # Create a list of chapter names for the dropdown
        chapter_names = st.session_state["toc"].apply(
            lambda row: f'{row["Chapter name"]} - {row["Chapter page"]} page', axis=1).tolist()
        #print(chapter_names)
        # Create a dropdown (select box) for chapter names
        st.session_state["selected_chapter"] = st.selectbox("Select a chapter", chapter_names)

        # Find the index of the selected chapter
        selected_index = chapter_names.index(st.session_state["selected_chapter"])
        update_selected_pages(selected_index)
        st.write(
            f"Selected Page Start: {st.session_state['num_start']}, Selected Page End: {st.session_state['num_end']}")

    option = st.selectbox('Choose type of summarization', ('Extractive (textrank)', 'Abstractive (BART)'), index=0)
    if 'textrank' in option:
        st.session_state['summary'] = TEXTRANK_SUMMARY
    elif 'BART' in option:
        st.session_state['summary'] = BART_SUMMARY
    else:
        st.session_state['summary'] = ERROR_SUMMARY

    # Input fields and variable/s
    col1, col2 = st.columns(2)

    with col1:
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


    with col2:
        st.session_state['whole_report'] = st.checkbox("Whole Report", False)
        # if st.button ("Custom page"):
        #     st.session_state['num_start'] = st.number_input("Start from page (0-indexed)", min_value=0, value=0, step=1, key='start_page')
        #     st.session_state['num_end'] = st.number_input("End at page (0-indexed)", min_value=0, value=5, step=1, key='end_page')

        # Custom keywords input
        st.write("\n\n")
        st.write("\n\n")
        st.write("\n\n")
        st.write("\n\n")
        st.write("\n\n")
        st.write("\n\n")
        st.write("\n\n")
        st.write("\n\n")
        custom_keywords = st.text_area("Enter custom keywords (separated by commas)")
        if custom_keywords:
            st.session_state['custom_keywords'] = [word.strip() for word in custom_keywords.split(',')]
            st.write(custom_keywords)
        if st.button("Clear keywords") and 'custom_keywords' in st.session_state:
            st.session_state['custom_keywords'] = []
            st.write("Custom keywords have been cleared.")
        # Processing the uploaded file (placeholder logic)
        if st.button('Run pipeline') and uploaded_file is not None:
            if st.session_state['whole_report']:
                st.session_state['num_end'] = None
                st.session_state['num_start'] = None
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
        #st.write(st.session_state['company_name'])
        col1, col2 = st.columns(2)

        with col1:
            st.subheader('Company Details')
            company = st.text_input('Company', st.session_state['company_name'])
            year = st.text_input('Year', st.session_state['report_year'])
            report_type = st.text_input('Type', st.session_state['report type'])
            pages_whole = st.number_input('Pages (whole report)', value=st.session_state['report_page_count'])
            if(st.session_state['whole_report'] == False):
                page_num_selected = st.session_state['num_end'] - st.session_state['num_start']
                pages_selected = st.number_input('Pages (selected part)', value=page_num_selected)
            avg_words = st.number_input('Average Words per Page (selected part)', value= st.session_state['Word_count']/page_num_selected)
            sentences = st.number_input('Number of Sentences (in the selected part)', value=len(st.session_state['preprocessed_df']))
            #250 is an average reading speed for an adult
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

def custom_slider_color(hex_color):
    st.markdown(f"""
        <style>
            .stSlider > div:nth-child(2) > div:nth-child(1) {{
                background: linear-gradient(to right, {hex_color} 0%, {hex_color} 100%);
            }}
        </style>
        """, unsafe_allow_html=True)

def show_advanced_analysis():
    # Advanced Analysis Tab Content
    st.header("Advanced Analysis")
    col1, col2 = st.columns(2)
    if 'bounded_report_text' in st.session_state:
        ESG_sentences = sa.extract_sentences_with_ngrams(sa.ESG_keywords, st.session_state['preprocessed_df'])
        Goals_sentences = sa.extract_sentences_with_keywords(sa.Goals_keywords, st.session_state['preprocessed_df'])
        Risk_factors  = sa.filter_sentences(st.session_state['preprocessed_df'], st.session_state["toc"], sa.Risk_chapter_pattern)
        Financial_state = sa.filter_sentences(st.session_state['preprocessed_df'], st.session_state["toc"], sa.Financial_chapter_pattern)
        columns_to_display = ["Original Sentence", "Sentiment Score"]

        with col1:
            #Financial state section
            st.markdown('<div class="financial-section">', unsafe_allow_html=True)
            st.subheader('Financial state')
            st.markdown('</div>', unsafe_allow_html=True)

            Financial_sentiment_df = sa.analyze_sentiment(Financial_state)
            Financial_sentiment_df_sel = Financial_sentiment_df[columns_to_display]
            st.write(Financial_sentiment_df_sel)
            st.session_state['average_sentiment_financial'] = Financial_sentiment_df["Sentiment Score"].mean()
            #st.write("Average sentiment score", st.session_state['average_sentiment_financial'])
            color_fin = front.get_slider_color(st.session_state['average_sentiment_financial'])
            st.markdown(
                f"Average sentiment score: <span style='color: {color_fin};'> {st.session_state['average_sentiment_financial']}</span>",
                unsafe_allow_html=True)
            #custom_slider_color(color_fin)
            sentiment_value = st.slider("Financial state", min_value=-1.0, max_value=1.0,
                                        value=float(st.session_state['average_sentiment_financial']), disabled=True)
            #ESG state section
            st.markdown('<div class="esg-section">', unsafe_allow_html=True)
            st.subheader('ESG')
            st.markdown('</div>', unsafe_allow_html=True)

            ESG_sentiment_df = sa.analyze_sentiment(ESG_sentences)
            ESG_sentiment_df_sel = ESG_sentiment_df[columns_to_display]
            st.write(ESG_sentiment_df_sel)

            st.session_state['average_sentiment_esg'] = ESG_sentiment_df["Sentiment Score"].mean()
            #st.write("Average sentiment score", st.session_state['average_sentiment_esg'])
            color_esg = front.get_slider_color(st.session_state['average_sentiment_esg'])
            # Use HTML to style the text
            st.markdown(f"Average sentiment score: <span style='color: {color_esg};'> {st.session_state['average_sentiment_esg']}</span>",
                        unsafe_allow_html=True)
            #custom_slider_color(color_esg)
            sentiment_value = st.slider("ESG Sentiment Score", min_value=-1.0, max_value=1.0,
                                            value=float(st.session_state['average_sentiment_esg']), disabled=True)
            #st.write(sentiment_value)

        with col2:
            # Goals and Objectives Section
            st.markdown('<div class="goals-section">', unsafe_allow_html=True)
            st.subheader('Goals and Objectives')
            st.markdown('</div>', unsafe_allow_html=True)

            Goals_sentiment_df = sa.analyze_sentiment(Goals_sentences)
            Goals_sentiment_df_sel = Goals_sentiment_df[columns_to_display]
            st.write(Goals_sentiment_df_sel)

            st.session_state['average_goals_sentiment'] = Goals_sentiment_df["Sentiment Score"].mean()
            #st.write("Average sentiment score", st.session_state['average_goals_sentiment'])
            color_goals = front.get_slider_color(sentiment_value)
            st.markdown(
                f"Average sentiment score: <span style='color: {color_goals};'> {st.session_state['average_goals_sentiment']}</span>",
                unsafe_allow_html=True)
            #custom_slider_color(color_goals)
            sentiment_value = st.slider("Goals and objectives Sentiment Score", min_value=-1.0, max_value=1.0,
                                        value=float(st.session_state['average_goals_sentiment']), disabled=True)

            # Risks and Threats Section
            st.markdown('<div class="risks-section">', unsafe_allow_html=True)
            st.subheader('Risks and Threats')
            st.markdown('</div>', unsafe_allow_html=True)

            Risks_sentiment_df = sa.analyze_sentiment(Risk_factors)
            Risks_sentiment_df_sel = Risks_sentiment_df[columns_to_display]
            st.write(Risks_sentiment_df_sel)

            st.session_state['average_risks_sentiment'] = Risks_sentiment_df["Sentiment Score"].mean()
            #st.write("Average sentiment score", st.session_state['average_risks_sentiment'])
            color_risk = front.get_slider_color(sentiment_value)
            st.markdown(
                f"Average sentiment score: <span style='color: {color_risk};'> {st.session_state['average_risks_sentiment']}</span>",
                unsafe_allow_html=True)
            #custom_slider_color(color_risk)
            color_html = f"<div style='width: 50px; height: 20px; background-color: {color_risk};'></div>"
            sentiment_value = st.slider("Goals and objectives Sentiment Score", min_value=-1.0, max_value=1.0,
                                        value=float(st.session_state['average_risks_sentiment']), disabled=True)

def show_table_extraction():
    
    if 'toc' in st.session_state:
        df = st.session_state['toc']
        df['End page'] = df['PDF index'].shift(-1, fill_value=df['PDF index'].iloc[-1]) # - 1
    else:
        df = pd.DataFrame()

    page_ranges = []
    col1, col2 = st.columns(2)
    with col1:
        for index, row in df.iterrows():
            chapter_label = f"{row['Chapter name']} (Page {row['Chapter page']})"
            if st.checkbox(chapter_label, key=index):
                page_ranges.append((row['PDF index'], row['End page']))
        # Extract Table Button at the bottom
    with col2:
        if st.button("Extract Table"):
            if 'uploaded_file' not in st.session_state:
                st.write('Please upload report')
            else:
                report_data = st.session_state['uploaded_file'].getvalue()
                if df.empty:
                    st.write(df)

                # TODO: make initialization once per session
                model_structure, model_detection, image_processor = tab.initializeTable()
                st.session_state['csv_strings'] = tab.TableExtractionFromStreamPageRanges(stream=report_data, keywords=tab.keywords,
                                                            page_ranges = page_ranges,                    #num_start=st.session_state['num_start'],num_end=st.session_state['num_end'],
                                                            model_detection = model_detection,
                                                            image_processor = image_processor
                                                            )
                n_table = 0
                print(st.session_state['csv_strings'])
                for i, csv_string in st.session_state['csv_strings']:
                    n_table = n_table + 1
                    st.write(f"Found on page {i}")
                    st.write(pd.read_csv(io.StringIO(csv_string),sep=',',header=None))
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
                            ["Main Page", "Additional Selection", "Summary Results", "Basic analysis", "Advanced Analysis", "Table Extraction" ])

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

