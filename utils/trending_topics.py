import os
import json
from datetime import datetime, timedelta
import pandas as pd

def get_trending_path():
    """
    Get the path to the trending topics file
    
    Returns:
        str: Path to the trending topics file
    """
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    os.makedirs(data_dir, exist_ok=True)
    return os.path.join(data_dir, "trending_topics.json")

def load_trending_topics():
    """
    Load trending topics from file
    
    Returns:
        dict: Dictionary of topic counts
    """
    trending_path = get_trending_path()
    if os.path.exists(trending_path):
        try:
            with open(trending_path, 'r') as f:
                data = json.load(f)
                # Convert string timestamps back to datetime objects
                if 'last_updated' in data:
                    data['last_updated'] = datetime.fromisoformat(data['last_updated'])
                return data
        except json.JSONDecodeError:
            # Return empty dict if file is corrupted
            return {'topics': {}, 'last_updated': datetime.now()}
    return {'topics': {}, 'last_updated': datetime.now()}

def save_trending_topics(trending_data):
    """
    Save trending topics to file
    
    Args:
        trending_data (dict): Dictionary with 'topics' and 'last_updated'
    """
    trending_path = get_trending_path()
    
    # Convert datetime to string for JSON serialization
    data_to_save = trending_data.copy()
    if 'last_updated' in data_to_save and isinstance(data_to_save['last_updated'], datetime):
        data_to_save['last_updated'] = data_to_save['last_updated'].isoformat()
    
    with open(trending_path, 'w') as f:
        json.dump(data_to_save, f)

def update_trending_topics(topics):
    """
    Update trending topics with new topics
    
    Args:
        topics (list): List of topic strings
        
    Returns:
        dict: Updated trending topics data
    """
    trending_data = load_trending_topics()
    
    # Reset trending if it's been more than a week
    if 'last_updated' in trending_data:
        if datetime.now() - trending_data['last_updated'] > timedelta(days=7):
            trending_data = {'topics': {}, 'last_updated': datetime.now()}
    
    # Update topic counts
    for topic in topics:
        if topic in trending_data['topics']:
            trending_data['topics'][topic] += 1
        else:
            trending_data['topics'][topic] = 1
    
    # Update last_updated timestamp
    trending_data['last_updated'] = datetime.now()
    
    # Save updated data
    save_trending_topics(trending_data)
    
    return trending_data

def get_top_trending_topics(limit=10):
    """
    Get the top trending topics
    
    Args:
        limit (int): Maximum number of topics to return
        
    Returns:
        list: List of (topic, count) tuples sorted by count
    """
    trending_data = load_trending_topics()
    
    # Sort topics by count
    sorted_topics = sorted(
        trending_data['topics'].items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    # Return top N topics
    return sorted_topics[:limit]

def get_trending_dataframe():
    """
    Get trending topics as a pandas DataFrame for visualization
    
    Returns:
        DataFrame: DataFrame with Topic and Count columns
    """
    trending_data = load_trending_topics()
    
    # Convert to DataFrame
    if trending_data['topics']:
        df = pd.DataFrame(
            sorted(trending_data['topics'].items(), key=lambda x: x[1], reverse=True),
            columns=['Topic', 'Count']
        )
        return df
    else:
        return pd.DataFrame(columns=['Topic', 'Count'])