import streamlit as st
import pandas as pd
import pickle
import re

# Text Preprocessor
class TextPreprocessor:
    @staticmethod
    def preprocess(text):
        if isinstance(text, str):
            text = text.lower()
            text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
            return text
        return ""

# Load the clustering model with caching
@st.cache_resource
def load_model():
    with open('job_clustering_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model()

st.title("Job Posting Clustering")

uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
    else:
        if 'job_description' not in df.columns:
            st.error("CSV must contain a 'job_description' column.")
        else:
            df['processed_text'] = df['job_description'].apply(TextPreprocessor.preprocess)

            try:
                # If your model expects raw text (like a pipeline with vectorizer), this works:
                clusters = model.predict(df['processed_text'])
                df['cluster'] = clusters
                st.write(df[['job_description', 'cluster']])
            except Exception as e:
                st.error(f"Error during clustering: {e}")
else:
    st.info("Please upload a CSV file containing job postings with a 'job_description' column.")
