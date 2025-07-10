# Employee Feedback Analysis

This Streamlit app allows HR teams to analyze employee feedback at scale using modern NLP models. It provides sentiment analysis, zero-shot topic classification, and generates summaries and actionable recommendations using LLMs (via Ollama or HuggingFace models).

## Features
- **Upload Excel (.xlsx) or CSV (.csv) files** containing employee feedback.
- **Sentiment analysis** of each comment (positive/neutral/negative).
- **Zero-shot topic classification**: Assigns each comment to an HR-related category.
- **Summary and recommendations**: For each topic, generates a concise summary and up to 3 actionable recommendations using a local LLM (Ollama, e.g., Llama2 or Mistral).
- **Downloadable results**: Export classification and summary results as Excel files.
- **Interactive charts**: Visualize sentiment and topic distributions.

## Requirements
- Python 3.8+
- pip (Python package manager)
- [Ollama](https://ollama.com/) installed and running locally (for LLM-based summarization/recommendation)
- Recommended: CUDA-compatible GPU for faster inference (optional)

### Python dependencies
Install all required packages with:
```bash
pip install -r requirements.txt
```
and install also torch framework manually 

## Usage
1. **Start Ollama** and ensure your desired LLM (e.g., llama2, mistral) is available locally.
2. **Run the Streamlit app:**
   ```bash
   streamlit run main.py
   ```
3. **Upload your feedback file** (Excel or CSV) via the web interface.
4. **Select the text column** containing employee comments.
5. **Click 'Analyze Sentiment and Classify'** to process the data.
6. **View results, download outputs, and generate summaries/recommendations.**

## File Format
- The uploaded file must contain at least one column with free-text employee feedback.
- Supported formats: `.xlsx`, `.csv`

## Customization
- You can modify the list of HR feedbacks categories in `main.py` (`candidate_labels`).
- To use a different LLM, change the model name in the summarization/recommendation functions.

## Example
![screenshot](screenshot.png)  <!-- Add a screenshot if available -->

## License
MIT License
