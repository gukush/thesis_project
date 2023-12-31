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
    # if value is None:
    #     return "gray"

        if value <= -0.05:
            return "red"
        elif value > -0.05 and value < 0.05:
            return "yellow"
        else:
            return "green"


css_style = """
    <style>
        .css-2trqyj, button.css-2trqyj  {
            padding: 0.25rem 0.5rem;  /* Smaller padding */
            margin: 5.125rem;  /* Reduced margin */
            font-size: 5.875rem;  /* Smaller font size */
        }
    </style>
"""


