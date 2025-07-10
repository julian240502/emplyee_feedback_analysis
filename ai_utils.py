import pandas as pd
import ollama
from transformers import pipeline
import torch
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


sentiment_pipeline= pipeline('sentiment-analysis', model = 'cardiffnlp/twitter-roberta-base-sentiment-latest', device=device)
classification_pipeline = pipeline('zero-shot-classification', model = 'knowledgator/comprehend_it-base', device=device)

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
    response = ollama.chat(
        model="gemma:2b",
        messages=[{"role": "user", "content": prompt}],
    )
    return response['message']['content'].strip()
    

def generate_recommendation(summary, topic):
    prompt = f"There is a summary of employee feedbacks about {topic} : {summary}\nYou are a Human resource expert, your job is to give maximum 3 recommendations about the summary."
    response = ollama.chat(
        model="gemma:2b",  # ou le nom exact de ton modèle Ollama local
        messages=[{"role": "user", "content": prompt}],
    )
    return response['message']['content'].strip()
