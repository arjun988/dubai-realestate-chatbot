import streamlit as st
import google.generativeai as genai
import pandas as pd
import os
from dotenv import load_dotenv
# Set page configuration
st.set_page_config(page_title="Dubai Real Estate Expert", layout="wide")

# Configure Gemini API

load_dotenv()

# Fetch the API key from the environment variable
fetched_api_key = os.getenv("gemini_gpt_key")

# Configure the generative AI with the fetched API key
genai.configure(api_key=fetched_api_key)

# Load and filter the dataframe
@st.cache_data
def load_data():
    df = pd.read_csv('trans_filter.csv')
    return df[['property_sub_type_en', 'property_usage_en', 'reg_type_en', 'area_name_en',
               'transaction_id', 'trans_group_en', 'procedure_name_en', 'project_name_en',
               'master_project_en', 'nearest_landmark_en', 'nearest_metro_en', 'nearest_mall_en',
               'rooms_en', 'has_parking', 'procedure_area', 'actual_worth', 'meter_sale_price',
               'rent_value', 'meter_rent_price', 'building_name_en', 'property_type_en',
               'instance_date']]

df_filter = load_data()

# Define system prompt
SYSTEM_PROMPT = """
You are an AI expert in Dubai real estate. Your role is to provide detailed and accurate information to users regarding the Dubai real estate market. You will answer queries related to properties, market trends, and historical data. Additionally, when asked, you will generate comprehensive investment probability reports for specific areas, considering the latest available data. Your responses should be up-to-date, data-driven, and provide actionable insights for potential investors.

You have access to a dataframe containing transactional data regarding real estate in Dubai with the following columns:
[List of columns and their descriptions...]

Respond as a real estate agent in a conversational manner rather than giving all the information at once.
"""

# Initialize Gemini model
model = genai.GenerativeModel('gemini-pro')

# Initialize chat in session state
if "chat" not in st.session_state:
    st.session_state.chat = model.start_chat(history=[])
    st.session_state.messages = []

# Chat interface
st.title("💬 Dubai Real Estate Expert")
st.write("Ask me anything about Dubai real estate market!")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
if prompt := st.chat_input("Ask your question about Dubai real estate"):
    # Display user message
    st.chat_message("user").write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get and display assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            # Get response from Gemini
            response = st.session_state.chat.send_message(
                f"{SYSTEM_PROMPT}\n\nUser Query: {prompt}",
                stream=True
            )
            
            # Stream the response
            for chunk in response:
                if chunk.text:
                    full_response += chunk.text
                    message_placeholder.write(full_response)
            
            # Add assistant's response to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Sidebar with additional features
with st.sidebar:
    st.header("📊 Market Statistics")
    
    # Add some quick statistics from the dataframe
    st.subheader("Quick Stats")
    st.write(f"Total Transactions: {len(df_filter):,}")
    st.write(f"Areas Covered: {df_filter['area_name_en'].nunique():,}")
    
    # Add filters
    st.subheader("Filters")
    selected_area = st.selectbox(
        "Select Area",
        options=["All"] + sorted(df_filter['area_name_en'].unique().tolist())
    )
    
    selected_property_type = st.selectbox(
        "Property Type",
        options=["All"] + sorted(df_filter['property_type_en'].unique().tolist())
    )
    
    # Show filtered statistics
    if selected_area != "All" or selected_property_type != "All":
        st.subheader("Filtered Stats")
        filtered_df = df_filter.copy()
        
        if selected_area != "All":
            filtered_df = filtered_df[filtered_df['area_name_en'] == selected_area]
        
        if selected_property_type != "All":
            filtered_df = filtered_df[filtered_df['property_type_en'] == selected_property_type]
        
        st.write(f"Number of Properties: {len(filtered_df):,}")
        if len(filtered_df) > 0:
            st.write(f"Average Price: {filtered_df['actual_worth'].mean():,.2f} AED")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit and Google's Gemini AI")