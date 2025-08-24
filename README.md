# Academic Research Co-Pilot
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://academic-research-copilot-yrewstulfwlne3u5ngww7a.streamlit.app/)

Academic Research Co-Pilot is an AI-powered Streamlit application that helps you quickly find, read, and summarize the most relevant academic papers for your research questions. It leverages advanced retrieval-augmented generation (RAG) techniques, semantic chunking, and large language models to deliver concise, well-structured reports based on the latest research.

## Features

- **Automated Paper Search:** Finds the most relevant papers from ArXiv based on your query.
- **Intelligent Selection:** Uses an LLM to select the single most relevant paper.
- **Semantic Reading:** Downloads, processes, and semantically chunks the paper for deep understanding.
- **Concise Reporting:** Generates a clear, referenced report answering your research question.
- **Source Attribution:** Every report includes the paper's title and direct URL.

## Project Structure

```
.
├── .env
├── .gitignore
├── app.py
├── requirements.txt
├── test_setup.py
└── src/
    ├── __init__.py
    ├── agent_core.py
    ├── rag_pipeline.py
    ├── tools.py
    └── __pycache__/
```

- [`app.py`](app.py): Streamlit app entry point.
- [`src/agent_core.py`](src/agent_core.py): Main agent workflow and orchestration.
- [`src/rag_pipeline.py`](src/rag_pipeline.py): PDF download, chunking, and retrieval logic.
- [`src/tools.py`](src/tools.py): ArXiv search tool.
- [`test_setup.py`](test_setup.py): Environment variable test script.
- [`.env`](.env): API keys and environment variables (not committed to version control).

## Setup

### 1. Clone the Repository

```sh
git clone <your-repo-url>
cd academic_research_copilot
```

### 2. Install Dependencies

It is recommended to use a virtual environment.

```sh
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in the project root with the following content:

```
GEMINI_API_KEY="your-gemini-api-key"
PINECONE_API_KEY="your-pinecone-api-key"
PINECONE_ENVIRONMENT="your-pinecone-environment"
```

Replace the values with your actual API keys.

### 4. Run the Application

```sh
streamlit run app.py
```

Open the provided URL in your browser to use the app.

## Usage

1. Enter your research question in the input box.
2. Click **Generate Report**.
3. Wait as the agent searches, selects, reads, and summarizes the most relevant paper.
4. Read the generated report, including the source citation.

## Testing

You can test environment variable loading with:

```sh
python test_setup.py
```

You can also run individual modules (e.g., [`src/tools.py`](src/tools.py), [`src/rag_pipeline.py`](src/rag_pipeline.py)) directly for debugging.

## Requirements

- Python 3.11+
- API keys for Gemini (Google Generative AI) and Pinecone

## License

This project is for academic and research purposes.

---

**Academic Research Co-Pilot** – Accelerate your literature review with AI!
