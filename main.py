import seaborn as sns
import pandas as pd
from transformers import pipeline
import streamlit as st
import io
import matplotlib.pyplot as plt
import ollama
import re
from ai_utils import sentiment_analysis, categories, summarize, generate_recommendation



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


# Streamlit app layout
st.title("Employee Feedback Analysis")

if 'result_df' not in st.session_state:
    st.session_state.result_df = None
if 'category_summaries' not in st.session_state:
    st.session_state.category_summaries = None

if 'already_displayed' not in st.session_state:
    st.session_state.already_displayed = False

uploaded_file = st.file_uploader("Upload your Excel or CSV file", type=["xlsx", "csv"])

if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
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
            if not isinstance(topic_counts, pd.Series):
                topic_counts = pd.Series(topic_counts)
            ax2.pie(topic_counts,
                    labels=topic_counts.index,
                    autopct='%1.1f%%', 
                    startangle=90, 
                    colors=plt.cm.Paired.colors,
                    textprops={'fontsize': 16})
            
            ax2.set_title("Category Distribution",fontsize=16)
            plt.tight_layout()
            st.pyplot(fig)

            # Utilisation de 'openpyxl' car compatible avec BytesIO pour pandas
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

            # Utilisation de 'openpyxl' car compatible avec BytesIO pour pandas
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