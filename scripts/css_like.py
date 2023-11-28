import streamlit as st

# Function to generate custom styles for different sections
def generate_custom_styles():
    st.markdown(
        """
        <style>
        .financial-section {
            background-color: #e8f4ea;  /* Light green background */
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 10px;
        }
        .esg-section {
            background-color: #e0f7fa;  /* Light blue background */
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 10px;
        }
        .goals-section {
            background-color: #fff3e0;  /* Light orange background */
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 10px;
        }
        .risks-section {
            background-color: #fce4ec;  /* Light pink background */
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Determine the color based on the sentiment value
def get_slider_color(value):
        if value < -0.5:
            return "red"
        elif value < 0.5:
            return "yellow"
        else:
            return "green"


