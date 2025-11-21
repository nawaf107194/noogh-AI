import requests
from bs4 import BeautifulSoup
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional

class WebSearchProvider:
    """
    Provides web search functionality using DuckDuckGo.
    Includes in-memory caching to avoid redundant searches.
    """
    def __init__(self, cache_ttl_hours: int = 24):
        """
        Initializes the WebSearchProvider.

        Args:
            cache_ttl_hours: Time-to-live for cache entries in hours.
        """
        self.cache_ttl = timedelta(hours=cache_ttl_hours)
        self._cache: Dict[str, Dict] = {}
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

    def _is_cache_valid(self, query: str) -> bool:
        """Checks if a cache entry for a query is still valid."""
        if query not in self._cache:
            return False
        
        entry_time = self._cache[query]["timestamp"]
        return (datetime.now() - entry_time) < self.cache_ttl

    def search(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """
        Performs a web search for the given query.

        Args:
            query: The search query.
            max_results: The maximum number of results to return.

        Returns:
            A list of dictionaries, where each dictionary represents a search result
            and contains 'title', 'link', and 'snippet'.
        """
        if self._is_cache_valid(query):
            print(f"WebSearchProvider: Returning cached results for '{query}'.")
            return self._cache[query]["results"][:max_results]

        print(f"WebSearchProvider: Performing new search for '{query}'.")
        
        try:
            response = requests.get(
                f"https://html.duckduckgo.com/html/?q={query}",
                headers=self.headers,
                timeout=10
            )
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            
            results = []
            result_nodes = soup.find_all("div", class_="result")

            for node in result_nodes:
                if len(results) >= max_results:
                    break

                title_node = node.find("a", class_="result__a")
                snippet_node = node.find("a", class_="result__snippet")
                link_node = node.find("a", class_="result__url")

                if title_node and snippet_node and link_node:
                    title = title_node.get_text(strip=True)
                    snippet = snippet_node.get_text(strip=True)
                    link = link_node["href"]
                    
                    # Clean up the link
                    if link.startswith("//"):
                        link = "https:" + link
                    
                    results.append({
                        "title": title,
                        "link": link,
                        "snippet": snippet
                    })

            self._cache[query] = {
                "timestamp": datetime.now(),
                "results": results
            }
            
            return results

        except requests.RequestException as e:
            print(f"WebSearchProvider: Error during web search: {e}")
            return []

if __name__ == '__main__':
    # Example usage
    search_provider = WebSearchProvider(cache_ttl_hours=1)
    
    # First search (will be slow and perform a real search)
    print("--- First search ---")
    search_results = search_provider.search("What is FastAPI?")
    for r in search_results:
        print(f"Title: {r['title']}")
        print(f"Link: {r['link']}")
        print(f"Snippet: {r['snippet']}\n")

    # Second search (should be fast and from cache)
    print("\n--- Second search (from cache) ---")
    search_results_cached = search_provider.search("What is FastAPI?")
    for r in search_results_cached:
        print(f"Title: {r['title']}")
        print(f"Snippet: {r['snippet']}\n")