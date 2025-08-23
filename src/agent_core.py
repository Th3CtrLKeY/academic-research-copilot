import os
import re
from typing import List, TypedDict

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph

# Import our previously created modules
from .rag_pipeline import (create_and_get_retriever,
                          download_pdf_from_arxiv, process_and_chunk_pdf)
from .tools import search_arxiv

# 1. Define the State for our agent 
class AgentState(TypedDict):
    question: str
    search_results: str
    selected_pdf_url: str
    paper_title: str  # NEW field for the paper's title
    retrieved_context: List
    report: str

# 2. Define the Nodes of our graph 

def search_papers_node(state: AgentState):
    """Node to search for academic papers."""
    print("--- Searching for papers ---")
    question = state["question"]
    results = search_arxiv.invoke(question)
    return {"search_results": results}

def select_best_paper_node(state: AgentState):
    """Node to analyze search results and select the most relevant paper."""
    print("--- Selecting best paper ---")
    question = state["question"]
    search_results = state["search_results"]

    prompt = f"""
    You are an expert research analyst. You have been given a user's question and a list of search results from ArXiv.
    Your task is to identify the single most relevant paper to answer the user's question.
    You must extract the title and the PDF URL of that single most relevant paper.
    
    Return the result as a single, clean line of text in the format:
    Title: | URL:

    User's Question: {question}

    Search Results:
    ---
    {search_results}
    ---
    """
    
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=gemini_api_key)
    response = llm.invoke(prompt)
    
    # Parse the LLM's response to extract title and URL
    response_text = response.content.strip()
    try:
        title_part, url_part = response_text.split(" | ")
        title = title_part.replace("Title: ", "").strip()
        url = url_part.replace("URL: ", "").strip()
    except ValueError:
        # Fallback in case the model doesn't follow the format perfectly
        title = "Title not found"
        url_match = re.search(r"https?://\S+", response_text)
        url = url_match.group(0) if url_match else "URL not found"

    return {"selected_pdf_url": url, "paper_title": title}

def process_paper_node(state: AgentState):
    """Node to download, process, and retrieve context from the selected paper."""
    print("--- Processing selected paper ---")
    pdf_url = state["selected_pdf_url"]
    question = state["question"]
    
    print(f"Processing URL: {pdf_url}")

    temp_pdf_path = "temp_paper.pdf"
    if not download_pdf_from_arxiv(pdf_url, temp_pdf_path):
        raise SystemExit("Failed to download PDF. Exiting.")

    paper_chunks = process_and_chunk_pdf(temp_pdf_path)
    
    pinecone_index_name = "academic-papers"
    retriever = create_and_get_retriever(paper_chunks, pinecone_index_name)
    
    retrieved_docs = retriever.invoke(question)
    
    os.remove(temp_pdf_path)
    
    return {"retrieved_context": retrieved_docs}

def generate_report_node(state: AgentState):
    """Node to generate the final report."""
    print("--- Generating final report ---")
    question = state["question"]
    context_docs = state["retrieved_context"]
    paper_title = state["paper_title"]
    pdf_url = state["selected_pdf_url"]
    
    context = "\n\n".join([doc.page_content for doc in context_docs])
    
    prompt = f"""
    You are a scientific research assistant. Your task is to generate a concise, well-structured report
    based on the user's question and the provided context from a research paper.

    User's Question: {question}

    Retrieved Context from the paper titled "{paper_title}":
    ---
    {context}
    ---

    Based ONLY on the context provided, generate a report that directly answers the user's question.
    At the end of the report, you MUST include a "Source" section with the paper's title and its URL.
    Do not include any "Note" add the end, make the changes that you deem are necessary.
    Format the source section exactly like this:
    
    **Source:**
    - **Title:** {paper_title}
    - **URL:** {pdf_url}
    """
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=gemini_api_key)
    response = llm.invoke(prompt)
    
    return {"report": response.content}

# 3. Wire the nodes together into a graph 
workflow = StateGraph(AgentState)

workflow.add_node("search_papers", search_papers_node)
workflow.add_node("select_best_paper", select_best_paper_node) 
workflow.add_node("process_paper", process_paper_node)
workflow.add_node("generate_report", generate_report_node)

workflow.set_entry_point("search_papers")
workflow.add_edge("search_papers", "select_best_paper")
workflow.add_edge("select_best_paper", "process_paper")
workflow.add_edge("process_paper", "generate_report")
workflow.add_edge("generate_report", END)

app = workflow.compile()

if __name__ == '__main__':
    load_dotenv()
    research_question = "What are the latest advancements in retrieval-augmented generation for medical diagnosis?"
    final_state = app.invoke({"question": research_question})
    print("\n\n--- FINAL REPORT ---")
    print(final_state["report"])