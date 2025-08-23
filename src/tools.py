import arxiv
from langchain_core.tools import tool

@tool
def search_arxiv(query: str) -> str:
    """
    Searches the ArXiv database for papers matching the query.
    Returns a formatted string of the top 5 papers with their title, ID, abstract, and PDF URL.
    """
    print(f"Searching ArXiv for: '{query}'")
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
        return f"An error occurred while searching ArXiv: {e}"

# This block allows us to test the tool directly
if __name__ == '__main__':
    print("--- Testing ArXiv Tool ---")
    test_query = "retrieval augmented generation"
    arxiv_results = search_arxiv.invoke(test_query)
    print(arxiv_results)