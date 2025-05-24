import streamlit as st
import pandas as pd
import pickle
from your_script_name import TextPreprocessor  # Replace with your actual script name if needed

# Load model
@st.cache_resource
def load_model():
    with open('job_clustering_model.pkl', 'rb') as f:
        data = pickle.load(f)
    return data['vectorizer'], data['model'], data['cluster_descriptions']

vectorizer, model, cluster_descriptions = load_model()

# Title
st.title("Job Recommendation Based on Skills")

# Input
user_skills = st.text_input("Enter your skills (comma separated)", "python, machine learning, statistics")

if st.button("Find Jobs"):
    if user_skills:
        # Preprocess and predict
        preprocessed = TextPreprocessor.preprocess(user_skills)
        vector = vectorizer.transform([preprocessed])
        cluster = model.predict(vector)[0]
        description = ", ".join(cluster_descriptions[cluster])

        st.success(f"Recommended Cluster: {cluster}")
        st.write(f"Top Skills in this cluster: {description}")

        # Load job data
        try:
            jobs_df = pd.read_csv("job_postings.csv")
            jobs_df['processed_skills'] = jobs_df['skills'].apply(TextPreprocessor.preprocess)
            jobs_df['cluster'] = model.predict(vectorizer.transform(jobs_df['processed_skills']))

            matched_jobs = jobs_df[jobs_df['cluster'] == cluster]
            st.subheader(f"Top Matching Jobs ({len(matched_jobs)})")
            st.dataframe(matched_jobs[['title', 'company', 'skills']].head(10))
        except Exception as e:
            st.error(f"Error loading jobs: {e}")
    else:
        st.warning("Please enter your skills.")
