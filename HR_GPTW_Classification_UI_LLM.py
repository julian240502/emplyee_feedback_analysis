import streamlit as st
import pandas as pd
import io
import openai
import torch
import time

# Check if CUDA (GPU) is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.write(f"Using device: {device}")

# Your API key and URL
# api_key = "m9b4wq+TMo4mv6U0ZWWas5KeCU3X7NAmnLfEjmp8XVg=" Pierre
api_key = "/TAQjyBklMetRPIWjUHbFV/SxelHKeWnbfqdZmxVF3A=" # key Julian
api_url = "http://px101.prod.exalead.com:8110/v1"

# Function to analyze sentiment and classify text using the Mistral API
def analyze_sentiment_and_classify(file_path, text_column, api_key, api_url, categories, sentiments):
    df = pd.read_excel(file_path)

    # Ensure the specified text column exists
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in the Excel file.")

    client = openai.OpenAI(api_key=api_key, base_url=api_url)
    results = []

    # Initialize progress bar
    progress_bar = st.progress(0)
    total_rows = len(df)
    step_text = st.empty()  # Placeholder for progress text

    for index, text in df[text_column].items():
        try:
            # Craft the template to include both sentiment and classification tasks
            template = f"""
            Please analyze the following text:
            1. Determine the sentiment of the text. The possible sentiments are: {', '.join(sentiments)}.
            2. Classify the text into one of the following categories: {', '.join(categories)}.

            Text:
            {text}

            You are HR Expert and manage multiple languages. Please provide the sentiment ({', '.join(sentiments)}) and the category label in the following format:
            Sentiment: [sentiment]
            Category: [category]
            """

            response = client.chat.completions.create(
                model="mistralai/Mistral-Small-24B-Instruct-2501",
                messages=[{"role": "user", "content": template}],
                max_tokens=1500,
                temperature=0
            )

            raw_response = response.choices[0].message.content.strip()
            parts = [line.strip() for line in raw_response.split("\n") if line.strip()]

            if len(parts) >= 2:
                sentiment_line = parts[-2]
                category_line = parts[-1]
                sentiment = sentiment_line.split(":")[1].strip().lower()
                category = category_line.split(":")[1].strip().split(" (")[0].strip().lower()
            else:
                sentiment = "neutral"
                category = "unknown"

            results.append({"sentiment": sentiment, "category": category})

            # Update progress bar and step text
            progress_bar.progress((index + 1) / total_rows)
            step_text.text(f"Step 2/5: Classifying rows... {index + 1}/{total_rows}")

            time.sleep(0.5)  # Delay for better UX

        except Exception as e:
            results.append({"sentiment": "error", "category": "error"})
            progress_bar.progress((index + 1) / total_rows)  # Update progress bar in case of an error
            step_text.text(f"Step 2/5: Classifying rows... {index + 1}/{total_rows} (Error)")

    df['sentiment'] = [result["sentiment"] for result in results]
    df['category'] = [result["category"] for result in results]
    return df


# Function to summarize feedback by category
def summarize_feedback_by_category(df, text_column, api_key, api_url):
    category_summaries = []
    client = openai.OpenAI(api_key=api_key, base_url=api_url)

    # Initialize progress bar for category processing
    progress_bar = st.progress(0)
    total_categories = len(df['category'].unique())
    step_text = st.empty()  # Placeholder for progress text

    for idx, (category, group) in enumerate(df.groupby('category')):
        combined_text = " ".join(group[text_column])

        template = f"""
        The following is a collection of feedback for the category: {category}. Please summarize the feedback.

        Feedback:
        {combined_text}

        Summary:
        """

        try:
            response = client.chat.completions.create(
                model="mistralai/Mistral-Small-24B-Instruct-2501",
                messages=[{"role": "user", "content": template}],
                max_tokens=300,
                temperature=0.5
            )

            if response.choices:
                summary = response.choices[0].message.content.strip()
                formatted_summary = f"{summary}"

                category_summaries.append({
                    "category": category,
                    "summary": formatted_summary
                })

            else:
                st.write(f"Error: No valid summary returned for category '{category}'.")

        except Exception as e:
            st.write(f"Error summarizing category '{category}': {e}")

        # Update progress bar and step text
        progress_bar.progress((idx + 1) / total_categories)
        step_text.text(f"Step 3/5: Summarizing categories... {idx + 1}/{total_categories}")

    return category_summaries


# Streamlit app layout
st.title("Employee Feedback Analysis")

# Initialize session state variables if they don't exist
if 'result_df' not in st.session_state:
    st.session_state.result_df = None
if 'category_summaries' not in st.session_state:
    st.session_state.category_summaries = None

# Step 1: Upload Excel file
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

if uploaded_file is not None:
    # Read the Excel file
    df = pd.read_excel(uploaded_file)

    if df.empty:
        st.error("Uploaded file is empty.")
    else:
        # Step 2: Select text column
        text_column = st.selectbox("Select the text column", df.columns)
        st.write(f"First few entries from the selected column `{text_column}`:")
        st.write(df[text_column].head())

        # Step 3: User inputs for sentiment and categories
        default_sentiments = ['negative', 'neutral', 'positive']
        default_categories = ['compensation', 'team interaction', 'work site', 'management', 'work life balance']

        sentiments_input = st.text_area("Enter sentiment labels", ', '.join(default_sentiments))
        sentiments = [s.strip().lower() for s in sentiments_input.split(',')]

        categories_input = st.text_area("Enter categories", ', '.join(default_categories))
        categories = [c.strip().lower() for c in categories_input.split(',')]

        # Step 4: Button to analyze sentiment and classify
        if st.button("Analyze Sentiment and Classify"):
            with st.spinner("Analyzing... Please wait."):
                st.session_state.result_df = analyze_sentiment_and_classify(uploaded_file, text_column, api_key, api_url, categories, sentiments)

        # Step 5: Display classification results
        if st.session_state.result_df is not None:
            st.write("### **Classification Results**")
            st.write(st.session_state.result_df)

            # Step 6: Download button for classification results after the table
            with st.container():
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

            # Step 7: Button to generate summaries
            if st.button("Generate Summaries"):
                with st.spinner("Processing... Please wait."):
                    st.session_state.category_summaries = summarize_feedback_by_category(st.session_state.result_df, text_column, api_key, api_url)

        # Step 8: Display summaries and enable download button for summaries
        if st.session_state.category_summaries is not None:
            st.write("### **Category Summaries**")
            for summary in st.session_state.category_summaries:
                st.write(f"**Category**: {summary['category']}")
                st.write(f"**Summary**: {summary['summary']}")
                st.write("---")

            # Display download button for summaries after the summaries are generated
            with st.container():
                output_summaries = io.BytesIO()
                summary_df = pd.DataFrame(st.session_state.category_summaries)
                with pd.ExcelWriter(output_summaries, engine='openpyxl') as writer:
                    summary_df.to_excel(writer, index=False)
                output_summaries.seek(0)

                st.download_button(
                    label="Download Summaries by Category",
                    data=output_summaries,
                    file_name="feedback_summaries_by_category.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
