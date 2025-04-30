import re
import unicodedata
from tqdm import tqdm

def clean_text(text):
    """
    Clean and normalize text content
    
    Args:
        text (str): The text to clean
        
    Returns:
        str: Cleaned text
    """
    if not text:
        return ""
    
    # Remove unicode characters
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove truncation markers often found in news API content
    text = re.sub(r'\[\+\d+ chars\]$', '', text)
    
    return text

def extract_main_content(article):
    """
    Extract and clean the main content from an article
    
    Args:
        article (dict): Article dictionary
        
    Returns:
        dict: Article with cleaned content
    """
    # Get content or fallback to description
    content = article.get("content", "")
    if not content:
        content = article.get("description", "")
    
    # Clean the content
    cleaned_content = clean_text(content)
    
    # Update the article
    updated_article = article.copy()
    updated_article["content"] = cleaned_content
    
    # Clean the title too
    if "title" in updated_article:
        updated_article["title"] = clean_text(updated_article["title"])
    
    return updated_article

def preprocess_articles(articles):
    """
    Preprocess a list of articles
    
    Args:
        articles (list): List of article dictionaries
        
    Returns:
        list: List of preprocessed article dictionaries
    """
    preprocessed_articles = []
    
    for article in tqdm(articles, desc="Preprocessing articles"):
        # Skip articles with no content
        if not article.get("content") and not article.get("description"):
            continue
            
        # Extract and clean the main content
        processed_article = extract_main_content(article)
        
        # Add to the list if it has content
        if processed_article["content"]:
            preprocessed_articles.append(processed_article)
    
    return preprocessed_articles

if __name__ == "__main__":
    # Test the preprocessing
    test_article = {
        "title": "Test Article Title",
        "content": "This is a test article content with some extra   spaces and unicode characters like ü and é. [+100 chars]",
        "source": "Test Source",
        "url": "https://example.com/test",
        "publishedAt": "2023-01-01T12:00:00Z"
    }
    
    processed = preprocess_articles([test_article])
    print("Original content:", test_article["content"])
    print("Processed content:", processed[0]["content"])