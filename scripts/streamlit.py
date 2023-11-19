import streamlit as st
import fitz
uploaded_file = st.file_uploader("Test upload")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    doc = fitz.Document(stream=bytes_data)
    for page in doc:
        st.write(len(page.get_text("text")))

st.write("Hello Worldhuhho!")
