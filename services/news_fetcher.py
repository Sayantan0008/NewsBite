import os
import requests
from datetime import datetime, timedelta

def fetch_news(days=1, language="en", page_size=100, from_date=None, to_date=None, query=None):
    """
    Fetch news articles from NewsAPI.org
    
    Args:
        days (int): Number of days to look back for articles (ignored if from_date and to_date are provided)
        language (str): Language of articles (e.g., 'en' for English)
        page_size (int): Number of articles to fetch (max 100 per request)
        from_date (str): Start date in YYYY-MM-DD format (overrides days parameter if provided)
        to_date (str): End date in YYYY-MM-DD format (required if from_date is provided)
        query (str): Custom search query (optional)
        
    Returns:
        list: List of article dictionaries
    """
    # Get API key from environment variable
    api_key = os.getenv("NEWS_API_KEY")
    if not api_key:
        raise ValueError("NEWS_API_KEY environment variable not set. Please set it with your NewsAPI.org API key.")
    
    # Calculate date range
    if from_date and to_date:
        # Use provided date range
        api_from_date = from_date
        api_to_date = to_date
    else:
        # Calculate based on days parameter
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Format dates for API
        api_from_date = start_date.strftime("%Y-%m-%d")
        api_to_date = end_date.strftime("%Y-%m-%d")
    
    # NewsAPI endpoint
    url = "https://newsapi.org/v2/everything"
    
    # Parameters for the API request
    params = {
        "apiKey": api_key,
        "q": query if query else "(technology OR artificial intelligence OR cybersecurity OR business OR finance OR economy OR politics OR government OR elections OR science OR research OR innovation OR health OR medical OR wellness OR entertainment OR movies OR music OR sports OR olympics OR world OR international OR global)",  # Required search query parameter
        "language": language,
        "from": api_from_date,
        "to": api_to_date,
        "pageSize": page_size,
        "sortBy": "publishedAt"
    }
    
    try:
        # Make the request
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise exception for HTTP errors
        
        # Parse response
        data = response.json()
        
        if data["status"] != "ok":
            print(f"Error from NewsAPI: {data.get('message', 'Unknown error')}")
            return []
        
        # Extract relevant fields from each article
        articles = []
        for article in data["articles"]:
            processed_article = {
                "title": article.get("title", ""),
                "content": article.get("content", "") or article.get("description", ""),
                "source": article.get("source", {}).get("name", "Unknown"),
                "url": article.get("url", ""),
                "publishedAt": article.get("publishedAt", "")
            }
            articles.append(processed_article)
        
        return articles
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching news: {e}")
        return []

if __name__ == "__main__":
    # Test the function
    articles = fetch_news()
    print(f"Fetched {len(articles)} articles")
    if articles:
        print("Sample article:")
        print(f"Title: {articles[0]['title']}")
        print(f"Source: {articles[0]['source']}")