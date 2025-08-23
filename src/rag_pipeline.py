import os
import re
import requests
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
load_dotenv()


def download_pdf_from_arxiv(arxiv_url: str, output_path: str) -> bool:
    """
    Downloads a PDF from an ArXiv URL.
    ArXiv URLs for PDFs are typically in the format: https://arxiv.org/pdf/2307.09288
    """
    try:
        response = requests.get(arxiv_url)
        response.raise_for_status()  # Raise an exception for bad status codes
        with open(output_path, 'wb') as f:
            f.write(response.content)
        print(f"Successfully downloaded PDF to {output_path}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error downloading PDF: {e}")
        return False

def process_and_chunk_pdf(pdf_path: str) -> list:
    """
    Loads a PDF and splits it into semantic chunks.
    """
    print(f"Loading PDF from {pdf_path}...")
    loader = UnstructuredPDFLoader(pdf_path)
    documents = loader.load()
    
    # The entire PDF is loaded as a list of documents. Concatenate their content.
    full_text = "\n".join(doc.page_content for doc in documents)
    
    print("Chunking document semantically...")
    # We use the Gemini embedding model for semantic chunking
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=gemini_api_key)
    
    text_splitter = SemanticChunker(
        embeddings, breakpoint_threshold_type="percentile"
    )
    
    chunks = text_splitter.create_documents([full_text])
    print(f"Document split into {len(chunks)} semantic chunks.")
    return chunks

def create_and_get_retriever(chunks: list, index_name: str):
    """
    Creates embeddings for document chunks and stores them in a Pinecone index.
    Returns a retriever object for the collection.
    """
    print(f"Creating and storing embeddings in index: {index_name}...")
    # Use the Gemini embedding model
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=gemini_api_key)
    
    # Store the chunks in Pinecone
    # This will create a new index if one doesn't exist, or use the existing one.
    vector_store = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=index_name
    )
    
    print("Embeddings stored successfully.")
    # Return a retriever object for querying
    return vector_store.as_retriever()

# This block allows us to test the RAG pipeline directly
if __name__ == '__main__':
    from dotenv import load_dotenv
    from tools import search_arxiv

    load_dotenv()

    # 1. Use our existing tool to find a paper
    print("--- Finding a test paper using ArXiv tool ---")
    query = "retrieval augmented generation"
    search_results = search_arxiv.invoke(query)

      
    # 2. Extract the PDF URL from the first result
    pdf_url_match = re.search(r"PDF URL: (https?://arxiv\.org/pdf/\S+)", search_results)
    if not pdf_url_match:
        raise ValueError("Could not find a PDF URL in the search results.")
    
    pdf_url = pdf_url_match.group(1)
    print(f"\n--- Found paper URL: {pdf_url} ---")
    
    # 3. Download the PDF
    temp_pdf_path = "temp_paper.pdf"
    if not download_pdf_from_arxiv(pdf_url, temp_pdf_path):
        raise SystemExit("Failed to download PDF. Exiting.")

    # 4. Process and chunk the PDF
    paper_chunks = process_and_chunk_pdf(temp_pdf_path)

    # 5. Use the Pinecone index we created earlier
    pinecone_index_name = "academic-papers"

    # 6. Store the chunks in our Pinecone vector database and get a retriever
    retriever = create_and_get_retriever(paper_chunks, pinecone_index_name)

    # 7. Test the retriever by asking a question
    print("\n--- Testing the retriever ---")
    test_question = "What is the main idea of this paper?"
    retrieved_chunks = retriever.invoke(test_question)
    
    print(f"\nQuery: {test_question}")
    print(f"Retrieved {len(retrieved_chunks)} chunks.")
    print("--- Top chunk content ---")
    if retrieved_chunks:
        print(retrieved_chunks[0].page_content)
    else:
        print("No chunks retrieved.")

    # 8. Clean up the downloaded file
    os.remove(temp_pdf_path)
    print(f"\nCleaned up {temp_pdf_path}")












