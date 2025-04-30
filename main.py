import os
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

# Apply patches to prevent PyTorch/Streamlit runtime errors
from utils.torch_streamlit_patch import apply_patches
apply_patches()

# Import project components
from services.news_fetcher import fetch_news
from model.summarizer import summarize_articles
from model.categorizer import categorize_articles
from utils.preprocessing import preprocess_articles
from utils.data_storage import save_compressed_data, optimize_article_content

# Load environment variables
load_dotenv()

def run_pipeline():
    """Run the complete NewsBite pipeline"""
    print("🔄 Starting NewsBite pipeline...")
    
    # Step 1: Fetch news articles
    print("📰 Fetching latest news articles...")
    articles = fetch_news()
    print(f"✅ Fetched {len(articles)} articles")
    
    if not articles:
        print("❌ No articles fetched. Exiting pipeline.")
        return
    
    # Step 2: Preprocess articles
    print("🔍 Preprocessing articles...")
    processed_articles = preprocess_articles(articles)
    
    # Step 3: Generate summaries
    print("📝 Generating article summaries...")
    articles_with_summaries = summarize_articles(processed_articles)
    
    # Step 4: Categorize articles
    print("🏷️ Categorizing articles...")
    categorized_articles = categorize_articles(articles_with_summaries)
    
    # Step 5: Optimize content length
    print("📏 Optimizing article content...")
    optimized_articles = optimize_article_content(categorized_articles)
    
    # Step 6: Save results with compression
    print("💾 Saving processed articles with compression...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"summarized_news_{timestamp}"
    
    # Save as compressed file
    output_path = save_compressed_data(optimized_articles, filename=filename, compress=True)
    print(f"✅ Saved {len(optimized_articles)} articles to {output_path}")
    
    return output_path

if __name__ == "__main__":
    output_file = run_pipeline()
    print(f"\n🎉 Pipeline completed! Results saved to {output_file}")
    print("\n🖥️ To view the dashboard, run: streamlit run app/dashboard.py")