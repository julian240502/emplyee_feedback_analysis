import seaborn as sns
import pandas as pd
from transformers import pipeline
import openai
import streamlit as st
import io
import matplotlib.pyplot as plt
import torch

import re


import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.write(f"Using device: {device}")

api_key = "/TAQjyBklMetRPIWjUHbFV/SxelHKeWnbfqdZmxVF3A=" # key Julian
api_url = "http://px101.prod.exalead.com:8110/v1"
client = openai.OpenAI(api_key=api_key, base_url=api_url)

sentiment_pipeline= pipeline('sentiment-analysis', model = 'cardiffnlp/twitter-roberta-base-sentiment-latest', device=device)
classification_pipeline = pipeline('zero-shot-classification', model = 'knowledgator/comprehend_it-base', device=device)

candidate_labels = [
    "3DS Product and customer focus",
    "Career growth & Development",
    "Collaboration & Teamwork",
    "Compensation & Benefits",
    "Engagement & Motivation",
    "Performance and accountability",
    "Processes & Organization",
    "Tools",
    "Working conditions & facilities",
    "Work-Life balance",
    "Workplace atmosphere & relationships",
    "Company vision, culture and strategy",
    "Leadership & Management",
    "Others"
]


def sentiment_analysis(comments, batch_size = 16):
    comments = comments[0].dropna().astype(str).str.strip().tolist()  # filter for empty comments
    comments = [c for c in comments if len(c) > 0]

    comments_df = pd.DataFrame(comments, columns=['comments'])

    results = []

    for i in range(0, len(comments_df), batch_size):
        batch = comments_df['comments'][i:i+batch_size].tolist()
        sentiments = sentiment_pipeline(batch, truncation=True, max_length=512)
        results.extend(sentiments)

    comments_df['sentiments'] = [res['label'] for res in results]
    return comments_df

def categories(comments_df, candidate_labels, batch_size=16):
    results = []
    texts = comments_df["comments"].astype(str).tolist()

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        outputs = classification_pipeline(
            batch,
            candidate_labels=candidate_labels,
            truncation=True
        )
        results.extend(outputs)

    comments_df = comments_df.copy()
    comments_df["Topic"] = [res["labels"][0] for res in results]
    return comments_df

def summarize(comments):
    prompt = f"""
You are a human resource expert.
You receive the following employee feedbacks related to a specific HR topic:

{ " ".join(comments) }

Write a concise and actionable summary (4-6 lines max) of these comments, highlighting the key points or concerns.
"""
    response = client.chat.completions.create(
        model="mistralai/Mistral-Small-24B-Instruct-2501",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0
    )
    return response.choices[0].message.content
    

def generate_recommendation(summary):
    prompt = f"There is a summary of employee feedbacks about {topic} : {summary}\n You are a Human ressource expert, your job is to give maximum 3 recommendations about the summary"
    response = client.chat.completions.create(
                model="mistralai/Mistral-Small-24B-Instruct-2501",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1500,
                temperature=0
            )
    return response.choices[0].message.content


# Streamlit app layout
st.title("Employee Feedback Analysis")

if 'result_df' not in st.session_state:
    st.session_state.result_df = None
if 'category_summaries' not in st.session_state:
    st.session_state.category_summaries = None

if 'already_displayed' not in st.session_state:
    st.session_state.already_displayed = False

uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)

    if df.empty:
        st.error("Uploaded file is empty.")
    else:
        text_column = st.selectbox("Select the text column", df.columns)

        if st.button("Analyze Sentiment and Classify (it takes 10 min)"):
            with st.spinner("Analyzing... Please wait."):
                comments_df = sentiment_analysis([df[text_column]])
                classified_df = categories(comments_df, candidate_labels)
                st.session_state.result_df = classified_df

        if st.session_state.result_df is not None:
            st.write("### **Classification Results**")
            st.write(st.session_state.result_df)

            # Distribution Charts
            st.write("### **Distributions**")

            sentiment_counts = st.session_state.result_df['sentiments'].value_counts()
            # Regroup the minor categories into "Others"
            topic_counts = st.session_state.result_df['Topic'].value_counts()
            top_n = 3
            already_has_others = "Others" in topic_counts.index

            if len(topic_counts) > top_n:
                top_categories = topic_counts[:top_n]

                if not already_has_others:
                    others_sum = topic_counts[top_n:].sum()
                    topic_counts = pd.concat([top_categories, pd.Series({'Others': others_sum})])
                else:
                    topic_counts = top_categories 
        
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

            #  Pie chart for sentiment distribution
            ax1.pie(sentiment_counts,
                    labels=sentiment_counts.index, 
                    autopct='%1.1f%%', 
                    startangle=90, 
                    colors=plt.cm.Paired.colors,
                    textprops={'fontsize': 16})
            
            ax1.set_title("Sentiment Distribution",fontsize=16)

            # Pie chart for category distribution
            ax2.pie(topic_counts,
                    labels=topic_counts.index,
                    autopct='%1.1f%%', 
                    startangle=90, 
                    colors=plt.cm.Paired.colors,
                    textprops={'fontsize': 16})
            
            ax2.set_title("Category Distribution",fontsize=16)
            plt.tight_layout()
            st.pyplot(fig)

            output_classification = io.BytesIO()
            with pd.ExcelWriter(output_classification, engine='openpyxl') as writer:
                st.session_state.result_df.to_excel(writer, index=False)
            output_classification.seek(0)

            st.download_button(
                label="Download Classification Results",
                data=output_classification,
                file_name="classification_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            if st.button("Generate Summaries"):
                with st.spinner("Generating summaries by topic..."):
                    summaries = []
                    categories = st.session_state.result_df['Topic'].unique()
                    progress_bar = st.progress(0)
                    total = len(categories)

                    for i, category in enumerate(categories):
                        if category.lower() == "others":
                            continue    #pass this category
                        subset = st.session_state.result_df[st.session_state.result_df['Topic'] == category]['comments']
                        summary = summarize(subset.tolist())
                        recommendation = generate_recommendation(summary, category)

                        st.markdown(f"### **Category**: {category}")
                        st.markdown(f"**Summary**: {summary}")
                        st.markdown(f"**Recommendation**: {recommendation}")
                        st.markdown("---")

                        summaries.append({
                            'category': category,
                            'summary': summary,
                            'recommendation': recommendation
                        })

                        progress_bar.progress((i + 1) / total)

                    st.session_state.category_summaries = summaries
                    st.session_state.already_displayed = True

        if st.session_state.category_summaries is not None and not st.session_state.already_displayed:
            st.write("### **Category Summaries and Recommendations**")
            for item in st.session_state.category_summaries:
                st.write(f"**Category**: {item['category']}")
                st.write(f"**Summary**: {item['summary']}")
                st.write(f"**Recommendation**:\n {item['recommendation']}")
                st.write("---")

            output_summaries = io.BytesIO()
            summary_df = pd.DataFrame(st.session_state.category_summaries)
            with pd.ExcelWriter(output_summaries, engine='openpyxl') as writer:
                summary_df.to_excel(writer, index=False)
            output_summaries.seek(0)

            st.download_button(
                label="Download Summaries and Recommendations",
                data=output_summaries,
                file_name="feedback_summaries_and_recommendations.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )