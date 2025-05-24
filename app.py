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

# Create tabs for different input methods
tab1, tab2 = st.tabs(["Upload CSV", "Enter Text"])

with tab1:
    st.header("Upload CSV File")
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'], key="file_uploader")
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
                    clusters = model.predict(df['processed_text'])
                    df['cluster'] = clusters
                    st.write("Clustering Results:")
                    st.write(df[['job_description', 'cluster']])
                    
                    # Add download button for results
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "Download Results",
                        csv,
                        "job_clustering_results.csv",
                        "text/csv",
                        key='download-csv'
                    )
                except Exception as e:
                    st.error(f"Error during clustering: {e}")
    else:
        st.info("Please upload a CSV file containing job postings with a 'job_description' column.")

with tab2:
    st.header("Enter Job Description")
    user_input = st.text_area("Paste the job description here:", height=200)
    
    if st.button("Classify Text"):
        if user_input.strip() == "":
            st.warning("Please enter a job description")
        else:
            processed_text = TextPreprocessor.preprocess(user_input)
            try:
                cluster = model.predict([processed_text])[0]
                st.write("## Classification Result")
                st.write(f"**Cluster:** {cluster}")
                st.write("**Processed Text:**")
                st.write(processed_text)
            except Exception as e:
                st.error(f"Error during classification: {e}")
