import arxiv
from langchain_core.tools import tool
from semanticscholar import SemanticScholar
import os
from dotenv import load_dotenv

load_dotenv()


@tool
def search_arxiv(query: str) -> str:
    """
    Searches the ArXiv database for papers matching the query.
    Returns a formatted string of the top 5 papers with their title, ID, abstract, and PDF URL.
    """
    print(f"Searching ArXiv for: {query}")
    try:
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=5,
            sort_by=arxiv.SortCriterion.Relevance
        )
        results = client.results(search)
        
        papers = []
        for r in results:
            papers.append(f"Title: {r.title}\nID: {r.entry_id}\nAbstract: {r.summary}\nPDF URL: {r.pdf_url}")
        
        if not papers:
            return "No results found on ArXiv."
            
        return "\n\n---\n\n".join(papers)
    except Exception as e:
        return f"Error searching ArXiv: {e}"

@tool
def search_semantic_scholar(query: str) -> str:
    """
    Searches Semantic Scholar for papers matching the query.
    Returns a formatted string of the top 5 papers with their title, ID, and abstract.
    """
    print(f"Searching Semantic Scholar for: {query}")
    try:
        ss_api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
        # Add a 10-second timeout to prevent hanging
        sch = SemanticScholar(api_key=ss_api_key, timeout=10) 
        results = sch.search_paper(query, limit=5)
        
        papers =
        for item in results:
            # Ensure abstract is not None before appending
            abstract = item.abstract if item.abstract else "No abstract available."
            papers.append(f"Title: {item.title}\nID: {item.paperId}\nAbstract: {abstract}")
            
        if not papers:
            return "No results found on Semantic Scholar."

        return "\n\n---\n\n".join(papers)
    except Exception as e:
        return f"Error searching Semantic Scholar: {e}"
# This block allows us to test the tools directly
if __name__ == '__main__':
    print("--- Testing ArXiv Tool ---")
    arxiv_results = search_arxiv.invoke("retrieval augmented generation")
    print(arxiv_results)

    print("\n\n--- Testing Semantic Scholar Tool ---")
    ss_results = search_semantic_scholar.invoke("retrieval augmented generation")
    print(ss_results)