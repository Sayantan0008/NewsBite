import os
import json
from datetime import datetime

def get_bookmark_path():
    """
    Get the path to the bookmarks file
    
    Returns:
        str: Path to the bookmarks file
    """
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    os.makedirs(data_dir, exist_ok=True)
    return os.path.join(data_dir, "bookmarks.json")

def load_bookmarks():
    """
    Load bookmarks from file
    
    Returns:
        list: List of bookmark dictionaries
    """
    bookmark_path = get_bookmark_path()
    if os.path.exists(bookmark_path):
        try:
            with open(bookmark_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            # Return empty list if file is corrupted
            return []
    return []

def save_bookmarks(bookmarks):
    """
    Save bookmarks to file
    
    Args:
        bookmarks (list): List of bookmark dictionaries
    """
    bookmark_path = get_bookmark_path()
    with open(bookmark_path, 'w') as f:
        json.dump(bookmarks, f)

def add_bookmark(article):
    """
    Add an article to bookmarks
    
    Args:
        article (dict): Article dictionary
        
    Returns:
        bool: True if added successfully, False if already exists
    """
    bookmarks = load_bookmarks()
    
    # Create a unique ID for the article
    article_id = f"{article.get('source', 'Unknown')}_{article.get('publishedAt', '')}_{article.get('title', '')[:20]}"
    
    # Check if already bookmarked
    if any(bookmark.get('id') == article_id for bookmark in bookmarks):
        return False
    
    # Create bookmark data
    bookmark_data = {
        'id': article_id,
        'title': article.get('title', ''),
        'source': article.get('source', 'Unknown'),
        'summary': article.get('summary', ''),
        'url': article.get('url', ''),
        'publishedAt': article.get('publishedAt', ''),
        'category': article.get('category', 'Uncategorized'),
        'bookmarked_at': datetime.now().isoformat()
    }
    
    # Add to bookmarks
    bookmarks.append(bookmark_data)
    save_bookmarks(bookmarks)
    return True

def remove_bookmark(article_id):
    """
    Remove an article from bookmarks
    
    Args:
        article_id (str): Unique ID of the article
        
    Returns:
        bool: True if removed successfully, False if not found
    """
    bookmarks = load_bookmarks()
    initial_count = len(bookmarks)
    
    # Filter out the bookmark with the given ID
    bookmarks = [b for b in bookmarks if b.get('id') != article_id]
    
    # Save if any bookmark was removed
    if len(bookmarks) < initial_count:
        save_bookmarks(bookmarks)
        return True
    return False

def get_bookmarks(limit=None, category=None):
    """
    Get bookmarks with optional filtering
    
    Args:
        limit (int, optional): Maximum number of bookmarks to return
        category (str, optional): Filter by category
        
    Returns:
        list: List of bookmark dictionaries
    """
    bookmarks = load_bookmarks()
    
    # Filter by category if specified
    if category:
        bookmarks = [b for b in bookmarks if b.get('category') == category]
    
    # Sort by bookmarked date (newest first)
    bookmarks.sort(key=lambda x: x.get('bookmarked_at', ''), reverse=True)
    
    # Limit results if specified
    if limit and isinstance(limit, int):
        bookmarks = bookmarks[:limit]
    
    return bookmarks