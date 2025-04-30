# Apply patching BEFORE any other imports to prevent runtime errors
import os
import sys
from dotenv import load_dotenv
load_dotenv()
# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import and apply our patching utility first thing
try:
    from utils.torch_streamlit_patch import apply_patches
    apply_patches()
    print("Applied PyTorch/Streamlit patches successfully")
except Exception as e:
    print(f"Warning: Could not apply patches to prevent runtime errors: {e}")

# Now it's safe to import other modules
import streamlit as st
import pandas as pd
import glob
import html
from datetime import datetime, timedelta
import json
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import time
import re

# Import project components
from services.news_fetcher import fetch_news
from utils.bookmark_manager import load_bookmarks, save_bookmarks, add_bookmark, remove_bookmark
from utils.trending_topics import load_trending_topics, update_trending_topics, get_trending_dataframe, get_top_trending_topics

# Set up exception handling for asyncio issues
import asyncio
try:
    asyncio.get_running_loop()
except RuntimeError:
    # No running event loop, create a new one
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    except Exception as e:
        # If we can't set up asyncio, we'll handle it gracefully in the app
        pass

# Set page configuration
st.set_page_config(
    page_title="NewsBite Dashboard",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for dark mode if it doesn't exist
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = True  # Set dark mode as default

# Apply custom CSS based on dark mode setting
def get_custom_css(dark_mode=True):
    """Return custom CSS based on dark/light mode preference"""
    
    if dark_mode:
        # Dark mode colors
        bg_color = "#111827"
        card_bg = "#1F2937"
        text_color = "#F9FAFB"
        muted_text = "#9CA3AF"
        primary_color = "#10B981"
        hover_color = "#059669"
        border_color = "#374151"
        accent_color = "#3B82F6"
        alert_color = "#F97316"  # Orange for alerts
    else:
        # Light mode colors
        bg_color = "#F9FAFB"
        card_bg = "#FFFFFF"
        text_color = "#111827"
        muted_text = "#6B7280"
        primary_color = "#10B981"
        hover_color = "#059669" 
        border_color = "#E5E7EB"
        accent_color = "#3B82F6"
        alert_color = "#F97316"  # Orange for alerts
    
    return f"""
    <style>
        :root {{
            --bg-color: {bg_color};
            --card-bg: {card_bg};
            --text-color: {text_color};
            --muted-text: {muted_text};
            --primary-color: {primary_color};
            --hover-color: {hover_color};
            --border-color: {border_color};
            --accent-color: {accent_color};
            --alert-color: {alert_color};
        }}
        
        /* Base styles */
        .stApp {{
            background-color: var(--bg-color);
            color: var(--text-color);
        }}
        
        /* Headings */
        h1, h2, h3, h4, h5, h6 {{
            color: var(--text-color);
            font-weight: 600;
            margin-bottom: 1rem;
        }}
        
        /* Article Cards */
        .article-card {{
            background-color: var(--card-bg);
            border-radius: 8px;
            border: 1px solid var(--border-color);
            padding: 1.5rem;
            margin-bottom: 1rem;
            transition: all 0.2s ease-in-out;
        }}
        
        .article-card:hover {{
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
            border-color: var(--primary-color);
        }}
        
        .article-title {{
            font-size: 1.25rem;
            font-weight: 600;
            margin-bottom: 0.75rem;
            line-height: 1.4;
        }}
        
        .article-meta {{
            display: flex;
            justify-content: space-between;
            color: var(--muted-text);
            font-size: 0.875rem;
            margin-bottom: 1rem;
        }}
        
        .article-summary {{
            font-size: 1rem;
            line-height: 1.6;
            margin-bottom: 1rem;
            padding: 0.5rem;
            border-radius: 4px;
            background-color: rgba(16, 185, 129, 0.05);
        }}
        
        /* News Alert Styles */
        .alert-summary {{
            padding: 0.75rem;
            background-color: rgba(249, 115, 22, 0.1);
            border-left: 3px solid var(--alert-color);
            border-radius: 4px;
            color: var(--text-color);
        }}
        
        /* Category Tags */
        .category-tag {{
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 500;
        }}
        
        /* Stats Cards */
        .stat-card {{
            background-color: var(--card-bg);
            border-radius: 8px;
            border: 1px solid var(--border-color);
            padding: 1rem;
            margin-bottom: 1rem;
            text-align: center;
        }}
        
        .stat-card h4 {{
            font-size: 1rem;
            margin-bottom: 0.5rem;
            color: var(--muted-text);
        }}
        
        /* Key points styling */
        .key-points-container {{
            margin: 15px 0;
            padding: 15px;
            border-radius: 8px;
            background-color: rgba(16, 185, 129, 0.1);
            border-left: 4px solid #10B981;
        }}
        
        .key-points-heading {{
            font-weight: 600;
            font-size: 18px;
            margin-bottom: 10px;
            color: var(--text-color);
        }}
        
        .key-points-list {{
            margin: 0;
            padding-left: 20px;
        }}
        
        .key-points-list li {{
            margin-bottom: 8px;
            color: var(--text-color);
            padding: 3px 0;
        }}
        
        .key-point-alert {{
            color: #e53e3e !important;
            font-weight: 500;
            padding: 3px 6px !important;
            background-color: rgba(229, 62, 62, 0.1);
            border-radius: 4px;
        }}
        
        /* Trending Items */
        .trending-item {{
            display: flex;
            justify-content: space-between;
            padding: 0.75rem;
            border-bottom: 1px solid var(--border-color);
            margin-bottom: 0.5rem;
        }}
        
        /* Button styles */
        .stButton button {{
            background-color: var(--card-bg);
            color: var(--text-color);
            border: 1px solid var(--border-color);
            transition: all 0.2s ease-in-out;
        }}
        
        .stButton button:hover {{
            border-color: var(--primary-color);
            color: var(--primary-color);
        }}
        
        /* Tab styling - Improved for better contrast in light mode */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 8px;
        }}
        
        .stTabs [data-baseweb="tab"] {{
            background-color: var(--card-bg);
            border-radius: 4px 4px 0 0;
            border: 1px solid var(--border-color);
            border-bottom: none;
            padding: 0.5rem 1rem;
            color: var(--text-color) !important;
        }}
        
        .stTabs [aria-selected="true"] {{
            background-color: var(--primary-color) !important;
            color: white !important;
        }}
        
        /* Fix StreamlitNavbar in light mode */
        .stApp [data-testid="stSidebarNav"] {{
            background-color: var(--card-bg) !important;
        }}
        
        .stApp [data-testid="stSidebarNav"] span {{
            color: var(--text-color) !important;
        }}
        
        /* Style the dashboard title in light mode */
        .stApp [data-testid="stAppViewContainer"] h1, 
        .stApp [data-testid="stAppViewContainer"] h2, 
        .stApp [data-testid="stAppViewContainer"] h3 {{
            color: var(--text-color) !important;
        }}
        
        /* Ensure tab text is always visible */
        .stApp .stTabs label {{
            color: var(--text-color) !important;
            font-weight: 500;
        }}
    </style>
    """

# Initialize session state for bookmarks and trending topics
if 'bookmarks' not in st.session_state:
    st.session_state.bookmarks = load_bookmarks()

if 'trending_topics' not in st.session_state:
    trending_data = load_trending_topics()
    st.session_state.trending_topics = trending_data.get('topics', {})

def load_latest_data():
    """Load the most recent news data file (supports .csv and .csv.gz)"""
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    csv_files = glob.glob(os.path.join(data_dir, "summarized_news_*.csv")) + \
                glob.glob(os.path.join(data_dir, "summarized_news_*.csv.gz"))

    if not csv_files:
        st.error("No news data found. Please run the pipeline first.")
        return None

    # Sort by filename (which contains timestamp)
    latest_file = sorted(csv_files)[-1]
    
    # Try multiple encodings to handle potential encoding issues
    encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'ISO-8859-1']
    
    for encoding in encodings_to_try:
        try:
            if latest_file.endswith('.gz'):
                return pd.read_csv(latest_file, compression='gzip', encoding=encoding)
            else:
                return pd.read_csv(latest_file, encoding=encoding)
        except UnicodeDecodeError:
            # If this encoding fails, try the next one
            continue
        except Exception as e:
            # For other errors, report and return None
            st.error(f"Error loading data file: {e}")
            return None
    
    # If all encodings fail
    st.error(f"Failed to decode file with any encoding: {latest_file}")
    return None

def format_date(date_str):
    """Format the date string for display"""
    try:
        if isinstance(date_str, pd.Timestamp):
            date = date_str
        else:
            date = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%SZ")
        return date.strftime("%b %d, %Y %H:%M")
    except:
        return date_str

def update_bookmarks():
    """Update bookmarks in session state and save to file"""
    save_bookmarks(st.session_state.bookmarks)

def toggle_theme():
    """Toggle between light and dark mode"""
    st.session_state.dark_mode = not st.session_state.dark_mode
    st.rerun()

def fetch_and_process_news():
    """Fetch news based on date range and process it"""
    from_date = st.session_state.from_date.strftime("%Y-%m-%d")
    to_date = st.session_state.to_date.strftime("%Y-%m-%d")
    
    with st.spinner("Fetching news articles..."):
        try:
            # Import necessary components
            from utils.preprocessing import preprocess_articles
            from utils.data_storage import save_compressed_data, optimize_article_content
            
            # Fetch news with date parameters
            articles = fetch_news(from_date=from_date, to_date=to_date)
            
            # Check if articles is a DataFrame (simulated data) or a list (API response)
            if isinstance(articles, pd.DataFrame):
                # If it's already a DataFrame, convert to list of dictionaries for consistent processing
                articles_list = articles.to_dict('records')
            else:
                # If it's already a list, use it directly
                articles_list = articles
            
            if not articles_list or len(articles_list) == 0:
                st.error("No articles found for the selected date range.")
                return
            
            # Process articles
            processed_articles = preprocess_articles(articles_list)
            
            # Safely import and use torch-dependent modules
            try:
                from model.summarizer import summarize_articles
                from model.categorizer import categorize_articles
                
                # For articles with minimal content, enhance them with title-based information
                for article in processed_articles:
                    if not article.get("content") or article.get("content", "").strip() == "":
                        if article.get("title"):
                            article["content"] = article.get("title", "")
                
                articles_with_summaries = summarize_articles(processed_articles)
                
                # Post-process summaries for missing/minimal content
                for article in articles_with_summaries:
                    # Check if summary indicates no content
                    if (article.get("summary") == "No content available for summarization." or 
                        not article.get("summary") or 
                        article.get("summary", "").strip() == ""):
                        
                        # Create better summary even in fallback mode
                        title = article.get("title", "")
                        content = article.get("content", "")
                        
                        if not content or content.strip() == "":
                            if title:
                                # Extract company name (usually first few words of title)
                                title_parts = title.split()
                                company_name = " ".join(title_parts[:2]) if len(title_parts) > 2 else title_parts[0] if title_parts else ""
                                
                                # Check if it's an earnings or financial report
                                if any(term in title.lower() for term in ["eps", "earnings", "reports", "results", "q1", "q2", "q3", "q4"]):
                                    # Create a more detailed summary with 3-4 sentences
                                    quarter_match = re.search(r'Q([1-4])', title)
                                    year_match = re.search(r'(20\d\d)', title)
                                    eps_match = re.search(r'EPS\s+(\S+)', title, re.IGNORECASE)
                                    
                                    quarter = quarter_match.group(0) if quarter_match else "quarterly"
                                    year = year_match.group(1) if year_match else "2025"
                                    eps_value = eps_match.group(1) if eps_match else "reported"
                                    
                                    # Extract any financial figures from the title
                                    figures = re.findall(r'\$?(\d+\.?\d*)\s?[mMbB]?', title)
                                    figures_text = ""
                                    if figures:
                                        figures_text = f" The report includes financial figures such as ${figures[0]}."
                                    
                                    consensus_match = re.search(r'consensus\s+(\S+)', title, re.IGNORECASE)
                                    consensus_text = ""
                                    if consensus_match:
                                        consensus_text = f" This compares to a consensus expectation of {consensus_match.group(1)}."
                                    
                                    article["summary"] = (
                                        f"{company_name} has published its {quarter} financial results for {year}. "
                                        f"The company reported earnings per share (EPS) of {eps_value}.{consensus_text}{figures_text} "
                                        f"Investors and analysts will be reviewing these results to assess the company's financial health and performance trends. "
                                        f"For detailed analysis of the complete financial performance, please refer to the full earnings report."
                                    )
                                elif "conference" in title.lower() or "call" in title.lower():
                                    article["summary"] = (
                                        f"{company_name} has scheduled a conference call or analyst event. "
                                        f"During this call, company executives will likely discuss recent performance, strategic initiatives, and future outlook. "
                                        f"Analysts and investors typically participate in these calls to gain deeper insights into the company's operations. "
                                        f"The information shared during this call may impact investment decisions and market perceptions of {company_name}."
                                    )
                                elif "initiated" in title.lower() or "rating" in title.lower():
                                    # Extract analyst firm
                                    firm_match = re.search(r'at\s+([A-Z][A-Za-z\s]+)', title)
                                    firm = firm_match.group(1) if firm_match else "a major investment firm"
                                    
                                    # Extract rating
                                    rating_match = re.search(r'with\s+(\w+)', title)
                                    rating = rating_match.group(1) if rating_match else "a new"
                                    
                                    article["summary"] = (
                                        f"{company_name} has received {rating} coverage from analysts at {firm}. "
                                        f"This new analyst rating suggests a positive outlook for the company's future performance. "
                                        f"Analyst ratings can significantly influence investor sentiment and stock price movements. "
                                        f"The full report likely contains a detailed analysis of the company's business model, competitive position, and growth prospects."
                                    )
                                else:
                                    article["summary"] = (
                                        f"Recent news has been published regarding {company_name}. "
                                        f"The article provides information about developments within the company or its industry. "
                                        f"These updates may be relevant for investors, customers, or industry observers tracking {company_name}. "
                                        f"For comprehensive details about this development, please refer to the complete article from the original source."
                                    )
                                
                                # Create more meaningful key points based on the type of article
                                if "eps" in title.lower() or "earnings" in title.lower():
                                    # Try to extract EPS figure and other financial metrics
                                    eps_match = re.search(r'EPS\s+(\S+)', title, re.IGNORECASE)
                                    eps_value = eps_match.group(1) if eps_match else "reported"
                                    
                                    consensus_match = re.search(r'consensus\s+(\S+)', title, re.IGNORECASE)
                                    consensus_value = consensus_match.group(1) if consensus_match else None
                                    
                                    article["key_points"] = [
                                        f"{company_name} has published its quarterly earnings report.",
                                        f"Earnings per share (EPS) reported at {eps_value}."
                                    ]
                                    
                                    if consensus_value:
                                        article["key_points"].append(f"Results compared to consensus expectations of {consensus_value}.")
                                        
                                    article["key_points"].append("Complete financial details are available in the full report.")
                                elif "conference" in title.lower() or "call" in title.lower():
                                    article["key_points"] = [
                                        f"{company_name} is hosting an analyst or investor conference call.",
                                        "Company executives will discuss performance and strategy.",
                                        "The call may reveal new information about future plans.",
                                        "Analysts and investors can gain insights beyond published reports."
                                    ]
                                elif "initiated" in title.lower() or "rating" in title.lower():
                                    # Extract analyst firm
                                    firm_match = re.search(r'at\s+([A-Z][A-Za-z\s]+)', title)
                                    firm = firm_match.group(1) if firm_match else "a major investment firm"
                                    
                                    # Extract rating
                                    rating_match = re.search(r'with\s+(\w+)', title)
                                    rating = rating_match.group(1) if rating_match else "a new"
                                    
                                    article["key_points"] = [
                                        f"{company_name} received {rating} coverage from {firm}.",
                                        "This represents a new analyst perspective on the company.",
                                        "Analyst ratings often impact investor sentiment and stock price.",
                                        "The full report contains detailed company analysis."
                                    ]
                                else:
                                    article["key_points"] = [
                                        f"News regarding {company_name} has been published.",
                                        "The article contains information about recent developments.",
                                        "This update may be significant for company stakeholders.",
                                        "Refer to the original source for comprehensive details."
                                    ]
                        elif content and len(content) > 200:
                            article["summary"] = content[:200] + "..."
                            # Extract key points different from summary
                            article["key_points"] = create_unique_key_points(0, content, title, article["summary"])
                        else:
                            article["summary"] = content
                            # Extract key points different from summary
                            article["key_points"] = create_unique_key_points(0, content, title, article["summary"])
                        
                        article["category"] = "Uncategorized"
                
                categorized_articles = categorize_articles(articles_with_summaries)
            except Exception as e:
                st.warning(f"AI model processing unavailable: {str(e)}. Using raw articles instead.")
                # Fallback: Use original articles without AI processing
                categorized_articles = processed_articles
                for article in categorized_articles:
                    # Create better summary even in fallback mode
                    title = article.get("title", "")
                    content = article.get("content", "")
                    
                    if not content or content.strip() == "":
                        if title:
                            # Extract company name (usually first few words of title)
                            title_parts = title.split()
                            company_name = " ".join(title_parts[:2]) if len(title_parts) > 2 else title_parts[0] if title_parts else ""
                            
                            # Check if it's an earnings or financial report
                            if any(term in title.lower() for term in ["eps", "earnings", "reports", "results", "q1", "q2", "q3", "q4"]):
                                # Create a more detailed summary with 3-4 sentences
                                quarter_match = re.search(r'Q([1-4])', title)
                                year_match = re.search(r'(20\d\d)', title)
                                eps_match = re.search(r'EPS\s+(\S+)', title, re.IGNORECASE)
                                
                                quarter = quarter_match.group(0) if quarter_match else "quarterly"
                                year = year_match.group(1) if year_match else "2025"
                                eps_value = eps_match.group(1) if eps_match else "reported"
                                
                                # Extract any financial figures from the title
                                figures = re.findall(r'\$?(\d+\.?\d*)\s?[mMbB]?', title)
                                figures_text = ""
                                if figures:
                                    figures_text = f" The report includes financial figures such as ${figures[0]}."
                                
                                consensus_match = re.search(r'consensus\s+(\S+)', title, re.IGNORECASE)
                                consensus_text = ""
                                if consensus_match:
                                    consensus_text = f" This compares to a consensus expectation of {consensus_match.group(1)}."
                                
                                article["summary"] = (
                                    f"{company_name} has published its {quarter} financial results for {year}. "
                                    f"The company reported earnings per share (EPS) of {eps_value}.{consensus_text}{figures_text} "
                                    f"Investors and analysts will be reviewing these results to assess the company's financial health and performance trends. "
                                    f"For detailed analysis of the complete financial performance, please refer to the full earnings report."
                                )
                            elif "conference" in title.lower() or "call" in title.lower():
                                article["summary"] = (
                                    f"{company_name} has scheduled a conference call or analyst event. "
                                    f"During this call, company executives will likely discuss recent performance, strategic initiatives, and future outlook. "
                                    f"Analysts and investors typically participate in these calls to gain deeper insights into the company's operations. "
                                    f"The information shared during this call may impact investment decisions and market perceptions of {company_name}."
                                )
                            elif "initiated" in title.lower() or "rating" in title.lower():
                                # Extract analyst firm
                                firm_match = re.search(r'at\s+([A-Z][A-Za-z\s]+)', title)
                                firm = firm_match.group(1) if firm_match else "a major investment firm"
                                
                                # Extract rating
                                rating_match = re.search(r'with\s+(\w+)', title)
                                rating = rating_match.group(1) if rating_match else "a new"
                                
                                article["summary"] = (
                                    f"{company_name} has received {rating} coverage from analysts at {firm}. "
                                    f"This new analyst rating suggests a positive outlook for the company's future performance. "
                                    f"Analyst ratings can significantly influence investor sentiment and stock price movements. "
                                    f"The full report likely contains a detailed analysis of the company's business model, competitive position, and growth prospects."
                                )
                            else:
                                article["summary"] = (
                                    f"Recent news has been published regarding {company_name}. "
                                    f"The article provides information about developments within the company or its industry. "
                                    f"These updates may be relevant for investors, customers, or industry observers tracking {company_name}. "
                                    f"For comprehensive details about this development, please refer to the complete article from the original source."
                                )
                            
                            # Create more meaningful key points based on the type of article
                            if "eps" in title.lower() or "earnings" in title.lower():
                                # Try to extract EPS figure and other financial metrics
                                eps_match = re.search(r'EPS\s+(\S+)', title, re.IGNORECASE)
                                eps_value = eps_match.group(1) if eps_match else "reported"
                                
                                consensus_match = re.search(r'consensus\s+(\S+)', title, re.IGNORECASE)
                                consensus_value = consensus_match.group(1) if consensus_match else None
                                
                                article["key_points"] = [
                                    f"{company_name} has published its quarterly earnings report.",
                                    f"Earnings per share (EPS) reported at {eps_value}."
                                ]
                                
                                if consensus_value:
                                    article["key_points"].append(f"Results compared to consensus expectations of {consensus_value}.")
                                    
                                article["key_points"].append("Complete financial details are available in the full report.")
                            elif "conference" in title.lower() or "call" in title.lower():
                                article["key_points"] = [
                                    f"{company_name} is hosting an analyst or investor conference call.",
                                    "Company executives will discuss performance and strategy.",
                                    "The call may reveal new information about future plans.",
                                    "Analysts and investors can gain insights beyond published reports."
                                ]
                            elif "initiated" in title.lower() or "rating" in title.lower():
                                # Extract analyst firm
                                firm_match = re.search(r'at\s+([A-Z][A-Za-z\s]+)', title)
                                firm = firm_match.group(1) if firm_match else "a major investment firm"
                                
                                # Extract rating
                                rating_match = re.search(r'with\s+(\w+)', title)
                                rating = rating_match.group(1) if rating_match else "a new"
                                
                                article["key_points"] = [
                                    f"{company_name} received {rating} coverage from {firm}.",
                                    "This represents a new analyst perspective on the company.",
                                    "Analyst ratings often impact investor sentiment and stock price.",
                                    "The full report contains detailed company analysis."
                                ]
                            else:
                                article["key_points"] = [
                                    f"News regarding {company_name} has been published.",
                                    "The article contains information about recent developments.",
                                    "This update may be significant for company stakeholders.",
                                    "Refer to the original source for comprehensive details."
                                ]
                        else:
                            article["summary"] = "No summary available."
                            article["key_points"] = ["No key points available."]
                    elif content and len(content) > 200:
                        article["summary"] = content[:200] + "..."
                        # Extract key points different from summary
                        article["key_points"] = create_unique_key_points(0, content, title, article["summary"])
                    else:
                        article["summary"] = content
                        # Extract key points different from summary
                        article["key_points"] = create_unique_key_points(0, content, title, article["summary"])
                    
                    article["category"] = "Uncategorized"
            
            optimized_articles = optimize_article_content(categorized_articles)
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"summarized_news_{timestamp}"
            save_compressed_data(optimized_articles, filename=filename, compress=True)
            
            # Refresh the page to show new data
            st.rerun()
        except Exception as e:
            st.error(f"Error processing news: {str(e)}")
            import traceback
            st.error(traceback.format_exc())

def extract_key_points(content, num_points=3):
    """Extract key points from article content as bullet points"""
    if not content or not isinstance(content, str):
        return ["No key points available"]
    
    # Split content into sentences
    sentences = re.split(r'(?<=[.!?])\s+', content)
    
    # Remove very short sentences
    valid_sentences = [s for s in sentences if len(s.split()) > 6]
    
    if len(valid_sentences) <= num_points:
        # If we don't have many sentences, just use what we have
        key_points = valid_sentences
    else:
        # Start with the first sentence (usually important in news)
        key_points = [valid_sentences[0]]
        
        # Find sentences with important indicators (numbers, names, quotes, etc.)
        important_sentences = []
        for s in valid_sentences[1:]:
            # Check for indicators of important information
            if (re.search(r'\d+%|\$\d+|\d+\s+(million|billion|trillion)', s) or  # Numbers, percentages, money
                re.search(r'".*?"', s) or  # Quotes
                re.search(r'(according to|said|reported|announced|confirmed)', s.lower())):  # Attributions
                important_sentences.append(s)
        
        # Take most important sentences first
        remaining_points = min(num_points - len(key_points), len(important_sentences))
        key_points.extend(important_sentences[:remaining_points])
        
        # If we still need more points, take from remaining sentences
        if len(key_points) < num_points:
            # Skip sentences we've already used
            remaining_sentences = [s for s in valid_sentences[1:] if s not in important_sentences]
            # Choose sentences from different parts of the article
            if remaining_sentences:
                indices = [len(remaining_sentences) // 2]  # Middle of article
                if len(remaining_sentences) > 3:
                    indices.append(len(remaining_sentences) * 3 // 4)  # 3/4 through article
                
                for idx in indices:
                    if len(key_points) < num_points and idx < len(remaining_sentences):
                        key_points.append(remaining_sentences[idx])
    
    # Clean up and format
    formatted_points = []
    for point in key_points:
        # Remove any leading/trailing whitespace and ensure proper punctuation
        clean_point = point.strip()
        if not clean_point.endswith('.'):
            clean_point += '.'
        if not clean_point[0].isupper() and len(clean_point) > 1:
            clean_point = clean_point[0].upper() + clean_point[1:]
        formatted_points.append(clean_point)
    
    return formatted_points if formatted_points else ["No key points available"]

def create_unique_key_points(idx, content, title, summary=None):
    """
    Create key points from the content that are distinct from the summary.
    Extract factual information, numbers, quotes, or important details.
    """
    # Sanitize inputs first
    content = sanitize_content(content) if content else ""
    title = sanitize_content(title) if title else ""
    summary = sanitize_content(summary) if summary else ""
    
    # For news alerts or minimal content
    if not content or content == title or len(content.strip()) < 30:
        if "EPS" in title or "reports" in title or ("Q" in title and any(q in title for q in ["Q1", "Q2", "Q3", "Q4"])):
            # Financial report format
            # Extract the numbers from the title for earnings reports
            numbers = re.findall(r'-?\$?\d+\.?\d*[cCmMbBkK]?', title)
            company = title.split()[0]  # First word is usually company name
            
            key_points = []
            
            # Basic point about the earnings
            key_points.append(f"{company} released their quarterly financial results.")
            
            # Add earnings data point if available
            if "EPS" in title and numbers:
                key_points.append(f"Earnings per share (EPS) reported at {numbers[0]}.")
                
                # Add comparison if available
                if "vs" in title and len(numbers) > 1:
                    key_points.append(f"This compares to {numbers[1]} in the same period last year.")
            
            return key_points
            
        if title:
            # For other types of news alerts
            return [f"Breaking news: {title}"]
        return ["No key points available"]
    
    # Split content into sentences
    sentences = re.split(r'(?<=[.!?])\s+', content)
    
    # Select distinct sentences for key points (avoid using exact sentences from summary)
    key_points = []
    
    # Try to understand what kind of article this is
    is_financial = any(term in content.lower() for term in ["earnings", "revenue", "profit", "eps", "quarter", "fiscal"])
    is_tech = any(term in content.lower() for term in ["technology", "software", "hardware", "app", "digital", "tech", "ai", "artificial intelligence"])
    is_political = any(term in content.lower() for term in ["government", "election", "vote", "president", "congress", "administration", "policy"])
    
    # Look for sentences with specific patterns that indicate important information
    # These patterns include numbers, percentages, quotes, or specific keywords
    for s in sentences:
        s = s.strip()
        # Skip too short sentences
        if len(s.split()) < 5:
            continue
            
        # Skip sentences that are too similar to the summary
        if summary and similarity_score(s, summary) > 0.5:
            continue
        
        # Different priorities for different article types
        if is_financial:
            # Prioritize financial numbers
            if re.search(r'\$\d+|\d+%|million|billion|revenue|profit|earnings|growth|decline', s.lower()):
                key_points.append(s)
                
        elif is_tech:
            # Prioritize tech details, features, launches
            if re.search(r'launch|feature|technology|application|platform|service|product|user', s.lower()):
                key_points.append(s)
                
        elif is_political:
            # Prioritize policy details, statements, impacts
            if re.search(r'policy|statement|impact|effect|announced|proposed|plan|initiative', s.lower()):
                key_points.append(s)
                
        else:
            # General priorities
            # Prioritize sentences with numbers or statistics
            if re.search(r'\d+%|\$\d+|\d+\s+(million|billion|trillion)|increased by|decreased by', s):
                key_points.append(s)
            
            # Look for quotes
            elif re.search(r'"[^"]+"|"[^"]+"', s):
                key_points.append(s)
                    
            # Look for sentences with key phrases indicating important information
            elif re.search(r'according to|announced|revealed|confirmed|launched|reported', s.lower()):
                key_points.append(s)
        
        # Exit if we have enough key points
        if len(key_points) >= 3:
            break
    
    # If we couldn't find enough key points with the patterns above,
    # select sentences with potential facts (but still avoid summary sentences)
    if len(key_points) < 2:
        # Get middle sentences as they often contain important details
        mid_point = len(sentences) // 2
        candidates = sentences[1:min(len(sentences), 10)]  # Skip first sentence (often in summary)
        
        for s in candidates:
            s = s.strip()
            # Skip too short sentences, already selected, or in summary
            if len(s.split()) < 5 or s in key_points or (summary and similarity_score(s, summary) > 0.5):
                continue
                
            key_points.append(s)
            if len(key_points) >= 3:
                break
    
    # If we still don't have enough key points, add general points from the content
    if len(key_points) < 1:
        if summary:
            # Extract a different sentence from the content than what's in the summary
            for s in sentences:
                if similarity_score(s, summary) < 0.5 and len(s.split()) >= 5:
                    key_points.append(s)
                    break
        else:
            # No summary, so just use the first sentence
            if sentences and len(sentences) > 0:
                key_points.append(sentences[0])
    
    # Format key points
    formatted_points = []
    for point in key_points:
        clean_point = point.strip()
        if not clean_point.endswith('.'):
            clean_point += '.'
        if not clean_point[0].isupper() and len(clean_point) > 1:
            clean_point = clean_point[0].upper() + clean_point[1:]
        formatted_points.append(clean_point)
    
    # If we still have no key points, use the title as a fallback
    if not formatted_points and title:
        return [f"Related to: {title}"]
    
    return formatted_points if formatted_points else ["No key points available"]

def sanitize_content(text):
    """Sanitize text by removing HTML tags and JavaScript"""
    if text is None:
        return ""
    
    # Convert to string if not already
    if not isinstance(text, str):
        try:
            text = str(text)
        except:
            return ""
    
    # Remove HTML tags and JavaScript
    text = re.sub(r'<[^>]*>', '', text)  # Remove HTML tags
    text = re.sub(r'\{[^}]*\}', '', text)  # Remove JavaScript inside curly braces
    text = re.sub(r'window\.open\([^)]*\)', '', text)  # Remove window.open calls
    text = re.sub(r"'_blank'", '', text)  # Remove _blank references
    text = re.sub(r'return false;', '', text)  # Remove return false statements
    text = re.sub(r'\s*\&gt;\s*', ' ', text)  # Remove &gt; 
    
    # Clean up any leftover mess
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
    text = text.strip()
    
    return text

def safe_html(text):
    """Safely escape text for HTML display"""
    if text is None:
        return ""
    
    # Convert non-string types to string
    if not isinstance(text, str):
        try:
            text = str(text)
        except:
            return ""
    
    # First sanitize any HTML/JavaScript in the content
    text = sanitize_content(text)
    
    # Then escape for HTML display
    return html.escape(text)

def get_subcategory(category, content, category_mapping):
    """Determine subcategory based on content and category"""
    if not category or not content:
        return None
    
    content_lower = str(content).lower()
    subcategories = category_mapping.get(category, [])
    
    for sub in subcategories:
        if sub.lower() in content_lower:
            return sub
    
    return None

def get_article_topics(content):
    """Extract topics from article content"""
    if not content:
        return []
    
    common_topics = {
        "economy": "💰 Economy",
        "market": "📈 Markets",
        "health": "🏥 Health",
        "covid": "🦠 COVID",
        "climate": "🌍 Climate",
        "tech": "💻 Technology",
        "election": "🗳️ Elections",
        "sport": "🏆 Sports",
        "ai": "🤖 AI",
        "crypto": "🪙 Cryptocurrency",
        "education": "🎓 Education",
        "energy": "⚡ Energy",
        "finance": "💵 Finance",
        "food": "🍽️ Food",
        "travel": "✈️ Travel",
        "war": "⚔️ Conflict",
        "science": "🔬 Science",
        "space": "🚀 Space"
    }
    
    content_lower = str(content).lower()
    found_topics = []
    
    for keyword, topic in common_topics.items():
        if keyword in content_lower:
            found_topics.append(topic)
            # Add to trending topics count (also updates session state)
            update_trending_topics([topic])
    
    return found_topics[:5]  # Limit to 5 topics

def render_summary(summary_text):
    """Properly render an article summary with styling"""
    if not summary_text or pd.isna(summary_text) or str(summary_text).strip() == '':
        # Create a fallback summary if not available
        st.markdown('<div class="article-summary">No summary available.</div>', unsafe_allow_html=True)
        return
    
    # Clean and prepare the summary
    summary = safe_html(summary_text)
    if not summary or summary.isspace():
        st.markdown('<div class="article-summary">No summary available.</div>', unsafe_allow_html=True)
        return
    
    # Render the summary with proper styling
    st.markdown(f'<div class="article-summary">{summary}</div>', unsafe_allow_html=True)

def render_key_points(content, is_alert=False):
    """
    Render key points with improved formatting and distinctions from summary.
    """
    # Validate content
    if not content:
        return '<div class="key-points-container"><div class="key-points-heading">Key Takeaways</div><div>No key points available</div></div>'
    
    # Handle non-list content or NaN values - safely check without using pd.isna on lists
    if isinstance(content, (float, int)) or (not isinstance(content, (list, tuple, str))):
        return '<div class="key-points-container"><div class="key-points-heading">Key Takeaways</div><div>No key points available</div></div>'
    
    # Handle list or string content
    if isinstance(content, str):
        if content == "No key points available":
            return f'<div class="key-points-container"><div class="key-points-heading">Key Takeaways</div><div>No key points available</div></div>'
        content = [content]  # Convert to list for consistent handling
    
    # Safety check for empty content 
    if not content or len(content) == 0:
        return '<div class="key-points-container"><div class="key-points-heading">Key Takeaways</div><div>No key points available</div></div>'
    
    # Start building the HTML
    html = '<div class="key-points-container">'
    html += '<div class="key-points-heading">Key Takeaways</div>'
    
    # If alert, use alert styling
    if is_alert:
        html += '<ol class="key-points-list">'
        for point in content:
            if point and point != "No key points available":
                html += f'<li class="key-point-alert">{point}</li>'
        html += '</ol>'
    else:
        # Create an ordered list of key points
        html += '<ol class="key-points-list">'
        
        # Filter and limit key points to display (max 3)
        filtered_points = []
        
        # Add unique points
        for point in content:
            if point and point != "No key points available":
                # Check if this point is too similar to already added points
                unique = True
                for added_point in filtered_points:
                    if similarity_score(point, added_point) > 0.7:
                        unique = False
                        break
                
                if unique:
                    filtered_points.append(point)
                    
                    # Format each point as list item
                    html += f'<li>{point}</li>'
            
            # Limit to 3 key points
            if len(filtered_points) >= 3:
                break
        
        html += '</ol>'
    
    # Close the container
    html += '</div>'
    
    return html

def similarity_score(text1, text2):
    """Calculate similarity between two text strings (simple implementation)"""
    if not text1 or not text2:
        return 0
    
    # Convert to lower case and split into words
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    # Calculate Jaccard similarity
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    return intersection / union if union > 0 else 0

def render_category_path(category, subcategory=None):
    """Render category and subcategory as a breadcrumb path"""
    if not category:
        return
    
    category = safe_html(category)
    
    st.markdown('<div class="category-path">', unsafe_allow_html=True)
    st.markdown(f'<span class="category-path-item">{category}</span>', unsafe_allow_html=True)
    
    if subcategory:
        subcategory = safe_html(subcategory)
        st.markdown('<span class="category-path-separator">›</span>', unsafe_allow_html=True)
        st.markdown(f'<span class="category-path-item">{subcategory}</span>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def main():
    # Initialize session state for bookmarks and trending topics
    if 'bookmarks' not in st.session_state:
        st.session_state.bookmarks = load_bookmarks()

    if 'trending_topics' not in st.session_state:
        trending_data = load_trending_topics()
        st.session_state.trending_topics = trending_data.get('topics', {})
    
    # Apply CSS based on dark mode setting
    st.markdown(get_custom_css(st.session_state.dark_mode), unsafe_allow_html=True)
    
    # Dark mode toggle in sidebar
    with st.sidebar:
        if st.checkbox("☀️ Light Mode", value=not st.session_state.dark_mode):
            st.session_state.dark_mode = False
        else:
            st.session_state.dark_mode = True
            
        if st.session_state.dark_mode != st.session_state.get('previous_mode', None):
            st.session_state.previous_mode = st.session_state.dark_mode
            st.rerun()  # Rerun to apply theme changes
    
    # Header
    st.title("📰 NewsBite Dashboard")
    st.markdown("### AI-powered news summarization and categorization")
    
    # Tabs for main content
    tab1, tab2, tab3 = st.tabs(["📰 News Feed", "🔖 Bookmarks", "📈 Trending Topics"])
    
    # Sidebar for filters and controls
    with st.sidebar:
        st.header("Fetch New Articles")
        col1, col2 = st.columns(2)
        
        with col1:
            default_from_date = datetime.now() - timedelta(days=7)
            from_date = st.date_input("From Date", default_from_date)
            st.session_state.from_date = from_date
        
        with col2:
            default_to_date = datetime.now()
            to_date = st.date_input("To Date", default_to_date)
            st.session_state.to_date = to_date
        
        if st.button("Fetch & Process News", use_container_width=True):
            fetch_and_process_news()
    
    # Load data
    df = load_latest_data()
    if df is None:
        st.stop()
    
    # Convert publishedAt to datetime for filtering
    if 'publishedAt' in df.columns:
        df['date'] = pd.to_datetime(df['publishedAt'], errors='coerce')
    
    # Define category mapping for subcategories
    category_mapping = {
        "Politics": ["International Relations", "Elections", "Legislation", "Government", "Policy"],
        "Business": ["Finance", "Economy", "Markets", "Startups", "Corporate", "Real Estate"],
        "Tech": ["AI", "Cybersecurity", "Software", "Hardware", "Startups", "Social Media"],
        "Health": ["Medical Research", "Healthcare Policy", "Wellness", "Disease", "Mental Health"],
        "Entertainment": ["Movies", "Music", "Celebrity", "TV Shows", "Gaming", "Arts"],
        "Sports": ["Football", "Basketball", "Tennis", "Olympics", "Motorsports", "Cricket"],
        "Science": ["Space", "Environment", "Research", "Climate", "Biology", "Physics"],
        "World": ["Europe", "Asia", "Americas", "Africa", "Middle East", "Oceania"]
    }
    
    # Define category colors with hex values
    category_color = {
        "Politics": "#FF6B6B",  # Red
        "Business": "#4ECDC4",  # Teal
        "Tech": "#7367F0",      # Purple
        "Health": "#28C76F",    # Green
        "Entertainment": "#FF9F43", # Orange
        "Sports": "#EA5455",    # Red-Orange
        "Science": "#1E9FF2",   # Blue
        "World": "#9F7AEA",     # Lavender
        "Uncategorized": "#6C757D"  # Gray
    }
    
    # Add subcategory to each article if not already present
    if 'subcategory' not in df.columns:
        df['subcategory'] = df.apply(
            lambda row: get_subcategory(
                row.get('category'), 
                row.get('content', ''), 
                category_mapping
            ),
            axis=1
        )
    
    # Add summaries if not already present
    if 'summary' not in df.columns or df['summary'].isna().any():
        df['summary'] = df.apply(
            lambda row: summarize_text(row.get('content', ''), max_length=200) 
            if pd.isna(row.get('summary', '')) or row.get('summary', '') == '' 
            else row.get('summary', ''),
            axis=1
        )
    
    # Dashboard Stats
    total_articles = len(df)
    unique_sources = len(df['source'].unique()) if 'source' in df.columns else 0
    categories_count = len(df['category'].unique()) if 'category' in df.columns else 0
    avg_reading_time = 0
    if 'content' in df.columns:
        word_counts = df['content'].apply(lambda x: len(str(x).split()) if x else 0)
        avg_reading_time = max(1, round(word_counts.mean() / 200))  # Avg reading speed: 200 words/min
    
    # Sidebar filters
    with st.sidebar:
        st.header("Filters")
        
        # Category filter
        if 'category' in df.columns:
            existing_categories = sorted(df['category'].unique().tolist())
            main_categories = ["All"] + existing_categories
            selected_main_category = st.selectbox("Category", main_categories)
            
            # Subcategory filter
            if selected_main_category != "All" and selected_main_category in category_mapping:
                subcategories = ["All"] + category_mapping.get(selected_main_category, [])
                selected_subcategory = st.selectbox("Subcategory", subcategories)
            else:
                selected_subcategory = "All"
        
        # Source filter
        if 'source' in df.columns:
            sources = ["All"] + sorted(df['source'].unique().tolist())
            selected_source = st.selectbox("Source", sources)
        
        # Date filter
        if 'date' in df.columns:
            min_date = df['date'].min().date()
            max_date = df['date'].max().date()
            selected_date = st.date_input(
                "Date Range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )
        
        # Search box
        search_term = st.text_input("Search articles", "")
    
    # Apply filters
    filtered_df = df.copy()
    
    # Category filter
    if 'category' in df.columns and selected_main_category != "All":
        filtered_df = filtered_df[filtered_df['category'] == selected_main_category]
        
        # Apply subcategory filter
        if selected_subcategory != "All" and 'subcategory' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['subcategory'] == selected_subcategory]
    
    # Source filter
    if 'source' in df.columns and selected_source != "All":
        filtered_df = filtered_df[filtered_df['source'] == selected_source]
    
    # Date filter
    if 'date' in df.columns and len(selected_date) == 2:
        start_date, end_date = selected_date
        filtered_df = filtered_df[
            (filtered_df['date'].dt.date >= start_date) &
            (filtered_df['date'].dt.date <= end_date)
        ]
    
    # Search filter
    if search_term:
        search_mask = (
            filtered_df['title'].str.contains(search_term, case=False, na=False) |
            filtered_df['summary'].str.contains(search_term, case=False, na=False) |
            filtered_df['content'].str.contains(search_term, case=False, na=False)
        )
        filtered_df = filtered_df[search_mask]
    
    # News Feed tab
    with tab1:
        # Display stats in a row
        st.markdown("#### Dashboard Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="stat-card">
                <h4>Articles</h4>
                <div style="font-size: 1.8rem; font-weight: 600;">{total_articles}</div>
                <div style="color: var(--muted-text);">Showing {len(filtered_df)}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
            <div class="stat-card">
                <h4>Sources</h4>
                <div style="font-size: 1.8rem; font-weight: 600;">{unique_sources}</div>
                <div style="color: var(--muted-text);">Unique publishers</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown(f"""
            <div class="stat-card">
                <h4>Categories</h4>
                <div style="font-size: 1.8rem; font-weight: 600;">{categories_count}</div>
                <div style="color: var(--muted-text);">Topic categories</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col4:
            st.markdown(f"""
            <div class="stat-card">
                <h4>Reading Time</h4>
                <div style="font-size: 1.8rem; font-weight: 600;">{avg_reading_time} min</div>
                <div style="color: var(--muted-text);">Average time</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Category distribution chart
        if 'category' in df.columns:
            st.markdown("#### Category Distribution")
            category_counts = filtered_df['category'].value_counts()
            st.bar_chart(category_counts)
        
        # Articles list
        st.markdown("#### Latest News Articles")
        
        for idx, row in filtered_df.iterrows():
            # Apply enhancements to summary and key points
            row = enhance_article_display(row)
            
            # Get category and color
            category = row.get('category', 'Uncategorized').strip()
            color = category_color.get(category, "#6B7280")  # Default gray for uncategorized
            
            # Get subcategory
            subcategory = row.get('subcategory')
            
            # Format date
            date_str = format_date(row.get('date', row.get('publishedAt', '')))
            
            # Extract key points using the index-based approach for uniqueness
            if not isinstance(row.get('key_points'), list) or not row.get('key_points') or (len(row.get('key_points', [])) == 1 and row.get('key_points', [''])[0].startswith("Related to:")):
                key_points = create_unique_key_points(
                    idx,  # Use the row index for unique patterns
                    row.get('content', ''),
                    row.get('title', ''),
                    row.get('summary', '')
                )
                row['key_points'] = key_points
            else:
                key_points = row.get('key_points', [])
            
            # Get topics
            topics = get_article_topics(row.get('content', ''))
            
            # Article card with improved styling
            st.markdown(f"""
            <div class="article-card">
                <div class="article-title">{safe_html(row['title'])}</div>
                <div class="article-meta">
                    <span>{safe_html(row.get('source', 'Unknown'))} • {date_str}</span>
                    <span>
                        <span class="category-tag" style="background-color: {color}20; color: {color}">
                            {safe_html(category)}
                        </span>
                        {f'<span class="category-tag" style="background-color: {color}10; color: {color}">{safe_html(subcategory)}</span>' if subcategory else ''}
                    </span>
                </div>
            """, unsafe_allow_html=True)
            
            # Display the summary - use our function that handles missing summaries
            summary_text = row.get('summary', '')
            if not summary_text or isinstance(summary_text, float) or (isinstance(summary_text, str) and str(summary_text).strip() == ''):
                # If no summary is available, generate one from content
                if row.get('content'):
                    summary_text = summarize_text(row.get('content', ''), max_length=200)
                else:
                    summary_text = "No summary available."
            
            # Ensure summary_text is a string
            if not isinstance(summary_text, str):
                summary_text = str(summary_text)
            
            st.markdown(f'<div class="article-summary">{safe_html(summary_text)}</div>', unsafe_allow_html=True)
            
            # Display key points directly under the summary
            if isinstance(key_points, list) and len(key_points) > 0 and isinstance(key_points[0], str) and key_points[0] != "No key points available" and not key_points[0].startswith("Related to:"):
                st.markdown(render_key_points(key_points, False), unsafe_allow_html=True)
            
            # Display article topics
            if topics:
                topics_html = ' '.join([f'<span style="margin-right:8px;">{topic}</span>' for topic in topics])
                st.markdown(f'<div style="margin-bottom:10px;">{topics_html}</div>', unsafe_allow_html=True)
            
            # Add "More Details" expander before the action buttons
            with st.expander("More Details"):
                tab_details, tab_full = st.tabs(["Article Details", "Full Content"])
                
                with tab_details:
                    cols = st.columns([2, 1])
                    with cols[0]:
                        st.markdown(f"**Source:** {safe_html(row.get('source', 'Unknown'))}")
                        st.markdown(f"**Published:** {date_str}")
                        st.markdown(f"**Category:** {safe_html(category)}{f' > {safe_html(subcategory)}' if subcategory else ''}")
                        
                        # Calculate reading time
                        if 'content' in row and row['content']:
                            word_count = len(str(row['content']).split())
                            reading_time = max(1, round(word_count / 200))
                            st.markdown(f"**Reading time:** {reading_time} min")
                    
                    with cols[1]:
                        # Topics based on content
                        if topics:
                            st.markdown("**Topics:**")
                            for topic in topics:
                                st.markdown(f"• {topic}")
                        
                        # URL if available
                        if 'url' in row and row['url']:
                            st.markdown(f"[Read full article]({row['url']})")
                
                with tab_full:
                    if 'content' in row and row['content']:
                        st.text_area("Full Content", value=row['content'], height=200, key=f"content_{idx}")
                    else:
                        st.info("No full content available for this article.")
            
            # Close the card
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Action buttons
            cols = st.columns([1, 1, 3])
            
            # Bookmark button
            article_id = f"{row.get('source', 'Unknown')}_{row.get('publishedAt', '')}_{row.get('title', '')[:20]}"
            is_bookmarked = any(bookmark.get('id') == article_id for bookmark in st.session_state.bookmarks)
            
            with cols[0]:
                if is_bookmarked:
                    if st.button("🔖 Remove", key=f"unbookmark_{idx}"):
                        st.session_state.bookmarks = [b for b in st.session_state.bookmarks if b.get('id') != article_id]
                        update_bookmarks()
                        st.rerun()
                else:
                    if st.button("🔖 Save", key=f"bookmark_{idx}"):
                        bookmark_data = {
                            'id': article_id,
                            'title': row.get('title', ''),
                            'source': row.get('source', 'Unknown'),
                            'summary': row.get('summary', ''),
                            'content': row.get('content', ''),
                            'url': row.get('url', ''),
                            'publishedAt': row.get('publishedAt', ''),
                            'category': row.get('category', 'Uncategorized'),
                            'subcategory': row.get('subcategory'),
                            'bookmarked_at': datetime.now().isoformat()
                        }
                        st.session_state.bookmarks.append(bookmark_data)
                        update_bookmarks()
                        st.success("Article bookmarked!")
            
            # Read more button
            with cols[1]:
                if 'url' in row and row['url']:
                    st.markdown(f"[Read full article]({row['url']})")
            
    # Bookmarks tab
    with tab2:
        st.markdown("#### Saved Articles")
        
        if not st.session_state.bookmarks:
            st.info("You haven't bookmarked any articles yet. Browse the News Feed tab and save articles you find interesting.")
        else:
            # Display bookmarks
            for idx, bookmark in enumerate(st.session_state.bookmarks):
                # Apply enhancements to bookmark summaries and key points
                bookmark = enhance_article_display(bookmark)
                
                category = bookmark.get('category', 'Uncategorized')
                color = category_color.get(category, "#6B7280")
                
                # Format date
                date_str = format_date(bookmark.get('publishedAt', bookmark.get('bookmarked_at', '')))
                
                # Extract key points
                if not isinstance(bookmark.get('key_points'), list) or not bookmark.get('key_points') or (len(bookmark.get('key_points', [])) == 1 and bookmark.get('key_points', [''])[0].startswith("Related to:")):
                    key_points = extract_key_points(bookmark.get('content', ''))
                    bookmark['key_points'] = key_points
                else:
                    key_points = bookmark.get('key_points', [])
                
                # Get subcategory
                subcategory = bookmark.get('subcategory')
                
                # Article card
                st.markdown(f"""
                <div class="article-card">
                    <div class="article-title">{safe_html(bookmark.get('title', 'Untitled'))}</div>
                    <div class="article-meta">
                        <span>{safe_html(bookmark.get('source', 'Unknown'))} • {date_str}</span>
                        <span>
                            <span class="category-tag" style="background-color: {color}20; color: {color}">
                                {safe_html(category)}
                            </span>
                            {f'<span class="category-tag" style="background-color: {color}10; color: {color}">{safe_html(subcategory)}</span>' if subcategory else ''}
                        </span>
                    </div>
                    <div class="article-summary">{safe_html(bookmark.get('summary', 'No summary available'))}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Display key points if available
                if isinstance(key_points, list) and len(key_points) > 0 and isinstance(key_points[0], str) and key_points[0] != "No key points available" and not key_points[0].startswith("Related to:"):
                    st.markdown(render_key_points(key_points, False), unsafe_allow_html=True)
                
                # Action buttons
                cols = st.columns([1, 1])
                
                with cols[0]:
                    if st.button("🗑️ Remove", key=f"remove_bookmark_{idx}"):
                        st.session_state.bookmarks.pop(idx)
                        update_bookmarks()
                        st.rerun()
                
                with cols[1]:
                    if bookmark.get('url'):
                        st.markdown(f"[Read full article]({bookmark['url']})")
    
    # Trending Topics tab
    with tab3:
        st.markdown("#### Trending Topics")
        
        trending_data = load_trending_topics()
        if not trending_data.get('topics'):
            st.info("No trending topics data available yet. Browse more articles to generate trending topics.")
        else:
            # Get trending topics as DataFrame
            topic_df = get_trending_dataframe()
            
            # Display as a chart
            st.bar_chart(topic_df.set_index('Topic'))
            
            # Display as a list
            st.markdown("#### Top Topics")
            
            top_topics = get_top_trending_topics(10)
            for topic, count in top_topics:
                st.markdown(f"""
                <div class="trending-item">
                    <span>{topic}</span>
                    <span style="font-weight: 600;">{count} articles</span>
                </div>
                """, unsafe_allow_html=True)
                
            # Show when trending data was last updated
            if 'last_updated' in trending_data:
                if isinstance(trending_data['last_updated'], datetime):
                    last_updated = trending_data['last_updated']
                else:
                    try:
                        last_updated = datetime.fromisoformat(trending_data['last_updated'])
                    except Exception:
                        last_updated = None
                
                if last_updated:
                    st.caption(f"Last updated: {last_updated.strftime('%Y-%m-%d %H:%M')}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>NewsBite - AI-powered news summarization and categorization</div>
        <div style="color: var(--muted-text); font-size: 0.8rem;">© 2023 NewsBite</div>
    </div>
    """, unsafe_allow_html=True)

# Function to fetch latest data file
def get_latest_data():
    """Get the path to the latest data file"""
    try:
        files = glob.glob("data/summarized_news_*.csv.gz")
        if not files:
            st.warning("No data files found. Please fetch some news first.")
            return None
        
        # Get the latest file by modification time
        latest_file = max(files, key=os.path.getmtime)
        return latest_file
    except Exception as e:
        st.error(f"Error finding data files: {e}")
        return None

# Function to load data from compressed file
def load_compressed_data(filename):
    """Load data from a compressed CSV file"""
    try:
        df = pd.read_csv(filename, compression='gzip')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# Function to save data to compressed file
def save_compressed_data(df, filename):
    """Save data to a compressed CSV file"""
    try:
        # Create data directory if it doesn't exist
        os.makedirs("data", exist_ok=True)
        
        # Save data
        filepath = os.path.join("data", filename)
        df.to_csv(filepath, compression='gzip', index=False)
        return True
    except Exception as e:
        st.error(f"Error saving data: {e}")
        return False

def fetch_news(from_date=None, to_date=None, keywords=None, sources=None, categories=None):
    """
    Fetch news from the NewsAPI
    Parameters:
    - from_date: Start date in YYYY-MM-DD format
    - to_date: End date in YYYY-MM-DD format
    - keywords: List of keywords to search for
    - sources: List of news sources to include
    - categories: List of categories to include
    """
    # Import the actual news fetcher from services
    from services.news_fetcher import fetch_news as api_fetch_news
    
    try:
        # Build query string if keywords provided
        query = None
        if keywords:
            if isinstance(keywords, list):
                query = " OR ".join(keywords)
            else:
                query = keywords
        
        # Call the actual API fetcher
        articles = api_fetch_news(
            from_date=from_date,
            to_date=to_date,
            query=query
        )
        
        # Sanitize content to remove HTML/JavaScript
        for article in articles:
            if article.get("title"):
                article["title"] = sanitize_content(article["title"])
            if article.get("content"):
                article["content"] = sanitize_content(article["content"])
            if article.get("summary"):
                article["summary"] = sanitize_content(article["summary"])
            if article.get("source"):
                article["source"] = sanitize_content(article["source"])
        
        # Ensure every article has at least minimal content for summarization
        for article in articles:
            # If content is missing, use title as content
            if not article.get("content") or article.get("content", "").strip() == "":
                if article.get("title"):
                    article["content"] = article.get("title", "")
        
        # If no articles returned, show a message
        if not articles:
            st.warning("No articles found for the selected date range.")
            return []
        
        return articles
    
    except Exception as e:
        st.error(f"Error fetching news from API: {str(e)}")
        return []

def get_category_colors():
    """Return a dictionary of category colors"""
    return {
        "Politics": "#FF6B6B",  # Red
        "Business": "#4ECDC4",  # Teal
        "Tech": "#7367F0",      # Purple
        "Health": "#28C76F",    # Green
        "Entertainment": "#FF9F43", # Orange
        "Sports": "#EA5455",    # Red-Orange
        "Science": "#1E9FF2",   # Blue
        "World": "#9F7AEA",     # Lavender
        "Uncategorized": "#6C757D"  # Gray
    }

def estimate_reading_time(text, words_per_minute=200):
    """Estimate reading time for text in minutes"""
    if not text or pd.isna(text):
        return 1  # Default minimum reading time
    
    # Count words
    word_count = len(re.findall(r'\w+', text))
    
    # Calculate reading time
    minutes = max(1, round(word_count / words_per_minute))
    
    return minutes

def summarize_text(text, max_length=150):
    """Create a better extractive summary of text by selecting key sentences"""
    if not text or pd.isna(text):
        return ""
    
    # If text is already short, just return it
    if len(text) <= max_length:
        return text
    
    # Split content into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Remove very short sentences
    valid_sentences = [s for s in sentences if len(s.split()) > 5]
    
    if not valid_sentences:
        # Fallback to simple truncation if we can't find valid sentences
        last_period = text[:max_length].rfind('.')
        if last_period > 0:
            return text[:last_period+1]
        return text[:max_length] + "..."
    
    # Always include the first sentence (usually contains the main point in news articles)
    summary_sentences = [valid_sentences[0]]
    current_length = len(summary_sentences[0])
    
    # Look for sentences with important information indicators
    important_indicators = [
        r'\d+%',                           # Percentages
        r'\$\d+',                          # Money amounts
        r'\d+\s+(million|billion)',        # Large numbers
        r'(increased|decreased|grew) by',  # Growth/decline metrics
        r'(announced|launched|revealed)',  # Announcements
        r'".*?"'                           # Quotes
    ]
    
    # Find important sentences from the middle of the article
    for s in valid_sentences[1:]:
        # Check if sentence has any important indicators
        is_important = any(re.search(pattern, s, re.IGNORECASE) for pattern in important_indicators)
        
        if is_important and current_length + len(s) + 1 <= max_length:
            summary_sentences.append(s)
            current_length += len(s) + 1  # +1 for space
            
            # If we have at least 3 sentences or reached 80% of max length, stop
            if len(summary_sentences) >= 3 or current_length >= 0.8 * max_length:
                break
    
    # If we couldn't find important sentences, just take the first few that fit
    if len(summary_sentences) == 1 and len(valid_sentences) > 1:
        for s in valid_sentences[1:3]:  # Try to add 2nd and 3rd sentences
            if current_length + len(s) + 1 <= max_length:
                summary_sentences.append(s)
                current_length += len(s) + 1
    
    # Join sentences into a summary
    summary = " ".join(summary_sentences)
    
    # Ensure the summary doesn't exceed max_length
    if len(summary) > max_length:
        last_period = summary[:max_length].rfind('.')
        if last_period > 0:
            summary = summary[:last_period+1]
        else:
            summary = summary[:max_length] + "..."
    
    return summary

def analyze_sentiment(text):
    """
    Analyze the sentiment of text
    This is a simple rule-based approach. In a real implementation,
    you would use a more sophisticated NLP model.
    """
    if not text or pd.isna(text):
        return 0
    
    # Lists of positive and negative words
    positive_words = ['good', 'great', 'excellent', 'positive', 'nice', 'wonderful', 'best', 
                      'success', 'successful', 'beneficial', 'valuable', 'happy', 'pleased', 
                      'impressive', 'remarkable', 'breakthrough', 'advance', 'improvement']
    
    negative_words = ['bad', 'poor', 'negative', 'terrible', 'worst', 'failure', 'difficult', 
                      'trouble', 'problem', 'issue', 'concern', 'risky', 'danger', 'harmful', 
                      'worry', 'disappointing', 'unfortunate', 'crisis', 'decline']
    
    # Convert to lowercase
    text = text.lower()
    
    # Count occurrences
    positive_count = sum(text.count(word) for word in positive_words)
    negative_count = sum(text.count(word) for word in negative_words)
    
    # Calculate sentiment score (-1 to 1)
    total = positive_count + negative_count
    if total == 0:
        return 0
    
    return (positive_count - negative_count) / total

def get_topics_from_text(text, min_topics=1, max_topics=3):
    """
    Extract key topics from text
    This is a simple keyword extraction approach. In a real implementation,
    you would use a more sophisticated topic modeling technique.
    """
    if not text or pd.isna(text):
        return []
    
    # Common topics/themes in news
    topics = {
        "Politics": ["government", "election", "president", "political", "policy", "vote", "democracy", "candidate"],
        "Economy": ["market", "economy", "financial", "business", "stock", "economic", "inflation", "trade"],
        "Technology": ["tech", "technology", "innovation", "digital", "software", "app", "startup", "AI", "artificial intelligence"],
        "Health": ["health", "medical", "covid", "vaccine", "healthcare", "patient", "disease", "pandemic"],
        "Environment": ["climate", "environment", "environmental", "green", "sustainable", "energy", "carbon", "renewable"],
        "Sports": ["game", "team", "player", "sport", "championship", "league", "win", "match", "tournament"],
        "Entertainment": ["movie", "film", "music", "celebrity", "entertainment", "actor", "actress", "show"],
        "Science": ["research", "study", "scientist", "scientific", "discovery", "space", "physics", "biology"]
    }
    
    text = text.lower()
    matches = {}
    
    # Count matches for each topic
    for topic, keywords in topics.items():
        count = sum(text.count(word) for word in keywords)
        if count > 0:
            matches[topic] = count
    
    # Sort by match count
    sorted_topics = sorted(matches.items(), key=lambda x: x[1], reverse=True)
    
    # Return top topics (between min and max)
    result = [topic for topic, _ in sorted_topics[:max_topics]]
    
    # If no topics found, return "General"
    if len(result) < min_topics:
        result.append("General")
    
    return result

def format_date(date_str):
    """Format date string for display"""
    if pd.isna(date_str):
        return ""
    
    try:
        date = pd.to_datetime(date_str)
        return date.strftime("%b %d, %Y")
    except:
        return date_str

def format_datetime(date_str):
    """Format datetime string for display"""
    if pd.isna(date_str):
        return ""
    
    try:
        date = pd.to_datetime(date_str)
        return date.strftime("%b %d, %Y %H:%M")
    except:
        return date_str

def enhance_article_display(row):
    """
    Enhance article display by improving summaries and key points that are too minimal
    This is applied at display time to ensure even existing database entries look good
    """
    title = row.get('title', '')
    content = row.get('content', '')
    summary = row.get('summary', '')
    key_points = row.get('key_points', [])
    
    # Handle cases where key_points might be a float (NaN)
    if isinstance(key_points, float) or not isinstance(key_points, (list, tuple)):
        key_points = []
        row['key_points'] = key_points
    
    # Check if summary needs enhancement (too short or contains placeholder text)
    needs_enhancement = (
        not summary or 
        isinstance(summary, float) or  # Handle NaN values
        (isinstance(summary, str) and (
            len(summary) < 60 or  # Too short
            "Earnings calls, analyst events" in summary or  # Generic placeholder
            summary.startswith("Related to:") or  # Generated placeholder
            "roadshows and more" in summary  # Generic placeholder
        ))
    )
    
    # Check if key points need enhancement
    key_points_need_enhancement = (
        not key_points or 
        len(key_points) == 0 or
        (len(key_points) == 1 and isinstance(key_points[0], str) and key_points[0].startswith("Related to:"))
    )
    
    # If we don't have actual content, we can only rely on the title
    if not content or len(content.strip()) < 100:
        if needs_enhancement:
            # Try to create a slightly more informative summary based on the title
            words = title.split()
            if len(words) > 5:
                # For longer titles, use the full title as is
                row['summary'] = title
            else:
                # For shorter titles, expand slightly
                category = row.get('category', '').capitalize()
                source = row.get('source', '')
                row['summary'] = f"{title}. {source} reports in the {category} category."
        
        if key_points_need_enhancement:
            # Create simple key points based on the title
            row['key_points'] = [
                title,
                row.get('source', 'Unknown source'),
                f"Published: {format_date(row.get('publishedAt', ''))}"
            ]
        return row
    
    # If we have enough content, proceed with smart enhancement
    if needs_enhancement or key_points_need_enhancement:
        # Enhanced summary based on article type
        if needs_enhancement:
            # First always try using our improved summarize_text function with content
            if content:
                extracted_summary = summarize_text(content, max_length=200)
                if extracted_summary and len(extracted_summary) > 40:
                    row['summary'] = extracted_summary
                    # Also generate key points if needed
                    if key_points_need_enhancement:
                        row['key_points'] = extract_key_points(content, num_points=3)
                    return row
            
            # If extraction failed or summary is still too short, use template
            if any(term in title.lower() for term in ["eps", "ffo", "earnings", "reports", "results", "q1", "q2", "q3", "q4"]):
                # Try to extract some financial details from content
                financial_details = ""
                if content:
                    # Look for revenue or profit numbers in content
                    revenue_match = re.search(r'revenue[s]? (?:of|was|were)\s+(\$?[\d\.]+\s+(?:million|billion|trillion))', content, re.IGNORECASE)
                    profit_match = re.search(r'(?:profit|income|earnings)[s]? (?:of|was|were)\s+(\$?[\d\.]+\s+(?:million|billion|trillion))', content, re.IGNORECASE)
                    
                    if revenue_match:
                        financial_details += f" Revenue was {revenue_match.group(1)}."
                    if profit_match:
                        financial_details += f" Profit reported as {profit_match.group(1)}."
                
                # Extract company name from title (usually first word or two)
                title_parts = title.split()
                company_name = " ".join(title_parts[:2]) if len(title_parts) > 2 else title_parts[0] if title_parts else ""
                
                # Financial/earnings reports
                quarter_match = re.search(r'Q([1-4])', title)
                year_match = re.search(r'(20\d\d)', title)
                
                # Extract financial metrics
                eps_match = re.search(r'EPS\s+(\S+)', title, re.IGNORECASE) or re.search(r'EPS view[^\d]+(\$?\d+\.?\d*)', title, re.IGNORECASE)
                ffo_match = re.search(r'FFO\s+(\S+)', title, re.IGNORECASE)
                
                financial_metric = None
                metric_name = "earnings"
                if eps_match:
                    financial_metric = eps_match.group(1)
                    metric_name = "earnings per share (EPS)"
                elif ffo_match:
                    financial_metric = ffo_match.group(1)
                    metric_name = "funds from operations (FFO)"
                
                quarter = quarter_match.group(0) if quarter_match else "quarterly"
                year = year_match.group(1) if year_match else "recent"
                
                # Extract consensus expectation
                consensus_match = re.search(r'consensus\s+(\$?\d+\.?\d*)', title, re.IGNORECASE)
                consensus_text = ""
                if consensus_match and financial_metric:
                    consensus_value = consensus_match.group(1)
                    financial_value = financial_metric.replace('$', '')
                    consensus_value_clean = consensus_value.replace('$', '')
                    try:
                        if float(financial_value) > float(consensus_value_clean):
                            consensus_text = f" This exceeds the consensus expectation of {consensus_value}."
                        elif float(financial_value) < float(consensus_value_clean):
                            consensus_text = f" This falls below the consensus expectation of {consensus_value}."
                        else:
                            consensus_text = f" This meets the consensus expectation of {consensus_value}."
                    except:
                        consensus_text = f" Analysts had projected {consensus_value}."
                
                # Try to extract more details from content
                outlook_text = ""
                guidance_match = re.search(r'guidance[^.]*?(increased|decreased|maintained|raised|lowered)', content, re.IGNORECASE)
                if guidance_match:
                    outlook_text = f" The company has {guidance_match.group(1)} its guidance."
                
                # Create comprehensive summary
                enhanced_summary = f"{company_name} has released {quarter} financial results"
                
                if year and year != "recent":
                    enhanced_summary += f" for {year}"
                
                enhanced_summary += "."
                
                if financial_metric:
                    enhanced_summary += f" The reported {metric_name} is {financial_metric}.{consensus_text}"
                
                # Add additional details extracted from content
                enhanced_summary += financial_details + outlook_text
                
                row['summary'] = enhanced_summary
            
            elif "index" in title.lower() or "spx" in title.lower() or "vix" in title.lower():
                # Market index summary - extract market movements from content
                market_movement = ""
                if content:
                    up_match = re.search(r'(?:gained|rose|climbed|increased|up)[^.]*?(\d+\.?\d*\s*%?|percent)', content, re.IGNORECASE)
                    down_match = re.search(r'(?:lost|fell|dropped|declined|down)[^.]*?(\d+\.?\d*\s*%?|percent)', content, re.IGNORECASE)
                    
                    if up_match:
                        market_movement = f" The index gained {up_match.group(1)}."
                    elif down_match:
                        market_movement = f" The index fell {down_match.group(1)}."
                
                # Extract index names
                index_names = []
                if "spx" in title.lower():
                    index_names.append("S&P 500")
                if "vix" in title.lower():
                    index_names.append("VIX Volatility Index")
                if "dow" in title.lower() or "djia" in title.lower():
                    index_names.append("Dow Jones")
                if "nasdaq" in title.lower():
                    index_names.append("NASDAQ")
                if not index_names:
                    index_names.append("market indices")
                
                indices = " and ".join(index_names)
                date_match = re.search(r'for ([A-Za-z]+ \d+)', title)
                date_str = date_match.group(1) if date_match else "recent trading"
                
                # Create informative summary with actual data points
                row['summary'] = f"This article covers {indices} performance for {date_str}.{market_movement}"
                
                # Extract additional market drivers if available
                if content:
                    cause_match = re.search(r'(due to|because of|following|amid|on)[^.]*?([^.]*)', content, re.IGNORECASE)
                    if cause_match:
                        row['summary'] += f" Market movement was {cause_match.group(1)} {cause_match.group(2)}."
            
            else:
                # For other articles, extract meaningful sentences from content
                if content and len(content) > 150:
                    # Break into sentences and find the most informative ones
                    sentences = re.split(r'(?<=[.!?])\s+', content)
                    valid_sentences = [s for s in sentences if len(s.split()) > 5 and len(s) < 150]
                    
                    if valid_sentences:
                        # Always include first sentence (often has main point)
                        summary = valid_sentences[0]
                        
                        # Look for a sentence with specifics (numbers, quotes, etc.)
                        for s in valid_sentences[1:]:
                            if re.search(r'\d+|\$|percent|said|announced|reported|launched', s.lower()):
                                if len(summary) + len(s) < 200:
                                    summary += " " + s
                                    break
                        
                        row['summary'] = summary
                    else:
                        # Fallback to title if no good sentences found
                        row['summary'] = title
                else:
                    # Not enough content, use title
                    row['summary'] = title

        # Enhance key points if needed
        if key_points_need_enhancement:
            # First try generating real key points from content
            if content and len(content) > 100:
                extracted_points = extract_key_points(content, num_points=3)
                if extracted_points and extracted_points[0] != "No key points available":
                    row['key_points'] = extracted_points
                    return row
            
            # If we still need key points, create them from title and metadata
            category = row.get('category', 'News')
            date_str = format_date(row.get('publishedAt', ''))
            source = row.get('source', 'Unknown source')
            
            # Try to extract a key data point from content
            data_point = ""
            if content:
                number_match = re.search(r'(\d+\.?\d*\s*(?:percent|%|million|billion|dollars|cents))', content)
                if number_match:
                    data_point = f"Data point: {number_match.group(1)}"
            
            key_points = [title]
            
            if date_str:
                key_points.append(f"Published: {date_str}")
            
            if data_point:
                key_points.append(data_point)
            else:
                key_points.append(f"Source: {source}")
            
            row['key_points'] = key_points
    
    return row

if __name__ == "__main__":
    main()