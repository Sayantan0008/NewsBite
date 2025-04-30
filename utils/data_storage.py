import os
import gzip
import pandas as pd
import json
from datetime import datetime

def save_compressed_data(data, filename=None, compress=True):
    """
    Save data to a compressed or uncompressed CSV file
    
    Args:
        data: List of dictionaries or DataFrame to save
        filename: Optional custom filename, otherwise uses timestamp
        compress: Whether to compress the file (default: True)
    
    Returns:
        Path to the saved file
    """
    # Convert to DataFrame if not already
    if not isinstance(data, pd.DataFrame):
        df = pd.DataFrame(data)
    else:
        df = data
    
    # Create timestamp for filename if not provided
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"summarized_news_{timestamp}"
    
    # Ensure data directory exists
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    os.makedirs(data_dir, exist_ok=True)
    
    # Determine file path based on compression setting
    if compress:
        file_path = os.path.join(data_dir, f"{filename}.csv.gz")
        with gzip.open(file_path, 'wt') as f:
            df.to_csv(f, index=False)
    else:
        file_path = os.path.join(data_dir, f"{filename}.csv")
        df.to_csv(file_path, index=False)
    
    return file_path

def optimize_article_content(articles, max_content_length=5000):
    """
    Optimize article content by truncating overly long articles
    
    Args:
        articles: List of article dictionaries or DataFrame
        max_content_length: Maximum length for article content
    
    Returns:
        Optimized articles
    """
    if isinstance(articles, pd.DataFrame):
        # Create a copy to avoid modifying the original
        df = articles.copy()
        
        # Truncate content if it exceeds max length
        if 'content' in df.columns:
            df['content'] = df['content'].apply(
                lambda x: x[:max_content_length] if isinstance(x, str) and len(x) > max_content_length else x
            )
        return df
    else:
        # For list of dictionaries
        optimized = []
        for article in articles:
            article_copy = article.copy()
            if 'content' in article_copy and isinstance(article_copy['content'], str):
                if len(article_copy['content']) > max_content_length:
                    article_copy['content'] = article_copy['content'][:max_content_length]
            optimized.append(article_copy)
        return optimized