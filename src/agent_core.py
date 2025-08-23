import os
import re
from typing import List, TypedDict

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph

# Import our previously created modules
from rag_pipeline import (create_and_get_retriever,
                          download_pdf_from_arxiv, process_and_chunk_pdf)
from tools import search_arxiv

# Define the state for our agent.
# This will be the "memory" that flows through the graph.
class AgentState(TypedDict):
    question: str
    search_results: str
    pdf_path: str
    retrieved_context: List
    report: str

# Define the nodes of our graph. Each node is a function that performs an action.

def search_papers_node(state: AgentState):
    """
    Node to search for academic papers using the ArXiv tool.
    """
    print("--- Searching for papers ---")
    question = state["question"]
    # Call the tool we created in Phase 2
    results = search_arxiv.invoke(question)
    return {"search_results": results}

def process_paper_node(state: AgentState):
    """
    Node to download, process, and retrieve context from the most relevant paper.
    """
    print("--- Processing selected paper ---")
    search_results = state["search_results"]
    question = state["question"]

    # As a simplifying step for this project, we'll process the *first* paper found.
    # A more advanced agent could ask the LLM to pick the best one from the list.
    pdf_url_match = re.search(r"PDF URL: (https?://arxiv\.org/pdf/\S+)", search_results)
    if not pdf_url_match:
        raise ValueError("Could not find a PDF URL in the search results.")
    
    pdf_url = pdf_url_match.group(1)
    print(f"Processing URL: {pdf_url}")

    # Use the RAG pipeline functions from Phase 3
    temp_pdf_path = "temp_paper.pdf"
    if not download_pdf_from_arxiv(pdf_url, temp_pdf_path):
        raise SystemExit("Failed to download PDF. Exiting.")

    paper_chunks = process_and_chunk_pdf(temp_pdf_path)
    
    # Use a unique name for the Pinecone index to avoid conflicts
    pinecone_index_name = "academic-papers"
    retriever = create_and_get_retriever(paper_chunks, pinecone_index_name)
    
    # Retrieve relevant context from the paper
    retrieved_docs = retriever.invoke(question)
    
    # Clean up the downloaded file
    os.remove(temp_pdf_path)
    
    return {"pdf_path": temp_pdf_path, "retrieved_context": retrieved_docs}

def generate_report_node(state: AgentState):
    """
    Node to generate the final report based on the retrieved context.
    """
    print("--- Generating final report ---")
    question = state["question"]
    context_docs = state["retrieved_context"]
    
    # Format the context for the LLM
    context = "\n\n".join([doc.page_content for doc in context_docs])
    
    prompt = f"""
    You are a scientific research assistant. Your task is to generate a concise, well-structured report
    based on the user's question and the provided context from a research paper.

    User's Question: {question}

    Retrieved Context from the paper:
    ---
    {context}
    ---

    Based on the context, please generate a report that directly answers the user's question.
    Ensure the report is clear, factual, and directly supported by the provided text.
    """
    
    # Initialize the Gemini model to act as the "brain"
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=gemini_api_key)
    
    # Generate the report
    response = llm.invoke(prompt)
    
    return {"report": response.content}

# Now, we wire the nodes together into a graph.
workflow = StateGraph(AgentState)

# Add the nodes
workflow.add_node("search_papers", search_papers_node)
workflow.add_node("process_paper", process_paper_node)
workflow.add_node("generate_report", generate_report_node)

# Set the entry point and define the sequence of operations
workflow.set_entry_point("search_papers")
workflow.add_edge("search_papers", "process_paper")
workflow.add_edge("process_paper", "generate_report")
workflow.add_edge("generate_report", END)

# Compile the graph into a runnable application
app = workflow.compile()

# This block allows us to test the agent directly
if __name__ == '__main__':
    load_dotenv()
    
    # Define the research question
    research_question = "What are the latest advancements LLMs?"
    
    # Run the agentic workflow
    final_state = app.invoke({"question": research_question})
    
    # Print the final report
    print("\n\n--- FINAL REPORT ---")
    print(final_state["report"])