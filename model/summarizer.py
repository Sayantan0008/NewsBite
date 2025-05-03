from transformers import PegasusForConditionalGeneration, PegasusTokenizer, pipeline
import torch
from torch.cuda.amp import autocast
from tqdm import tqdm
import re
import warnings
import os
import tensorflow as tf
import time
import concurrent.futures
import streamlit as st

# Configure tqdm to work well with Streamlit
tqdm.pandas()
# Set tqdm format for better display in different environments
tqdm_format = '{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=all, 1=info, 2=warning, 3=error
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Initialize CUDA if available
if torch.cuda.is_available():
    # Clean up GPU memory at start
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

# Global variables to avoid reloading model
MODEL = None
TOKENIZER = None

def verify_cuda_usage(silent=False):
    """
    Verify that CUDA is being properly used and display GPU memory usage
    
    Args:
        silent (bool): If True, suppress print messages (for use with progress bars)
    
    Returns:
        bool: True if CUDA is available and working, False otherwise
    """
    if torch.cuda.is_available():
        # Create a small test tensor and verify it's on CUDA
        test_tensor = torch.tensor([1.0, 2.0, 3.0])
        test_tensor = test_tensor.to("cuda")
        
        # Display GPU memory usage if not silent
        if not silent:
            print(f"Test tensor device: {test_tensor.device}")
            allocated = torch.cuda.memory_allocated(0) / 1024**2
            reserved = torch.cuda.memory_reserved(0) / 1024**2
            print(f"GPU Memory: {allocated:.2f}MB allocated, {reserved:.2f}MB reserved")
        return True
    else:
        if not silent:
            print("CUDA is not available. Using CPU for inference (slower).")
        return False

def load_model():
    """
    Load the Pegasus summarization model and tokenizer with half-precision for faster inference
    """
    global MODEL, TOKENIZER
    
    if MODEL is None or TOKENIZER is None:
        # Create a progress bar for model loading
        with tqdm(total=100, desc="Loading model", unit="%", bar_format=tqdm_format) as pbar:
            model_name = "google/pegasus-xsum"
            
            try:
                # Clean up GPU memory before loading model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Load tokenizer
                pbar.update(20)
                TOKENIZER = PegasusTokenizer.from_pretrained(model_name)
                
                # Load model
                pbar.update(40)
                MODEL = PegasusForConditionalGeneration.from_pretrained(model_name)
                
                # Move to GPU if available with half precision for faster inference and less VRAM usage
                if torch.cuda.is_available():
                    pbar.update(20)
                    MODEL = MODEL.half()  # Convert to half precision (FP16)
                    MODEL = MODEL.to("cuda")
                    # Verify CUDA usage silently without prints
                    if verify_cuda_usage(silent=True):
                        pbar.update(20)
                else:
                    pbar.update(40)  # Skip CUDA steps
            except Exception as e:
                # Create dummy model and tokenizer if loading fails
                if MODEL is None or TOKENIZER is None:
                    pbar.set_description("Using fallback summarization")
    
    return MODEL, TOKENIZER

def generate_summary(text, model, tokenizer, max_length=120, min_length=50):
    """
    Generate an abstractive summary for a single text
    Args:
        text (str): The text to summarize
        model: The pre-loaded summarization model
        tokenizer: The pre-loaded tokenizer
        max_length (int): Maximum length of the summary (in tokens)
        min_length (int): Minimum length of the summary (in tokens)
    Returns:
        str: The generated summary that ends with a period
    """
    # Skip empty or very short texts
    if not text or len(text.split()) < 10:
        return ""

    # Truncate text if it's too long (Pegasus has a limit)
    max_input_length = tokenizer.model_max_length
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_input_length)
    
    # Move inputs to GPU if available
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    # Calculate appropriate summary length based on input text length
    text_word_count = len(text.split())
    if text_word_count > 500:
        adjusted_max_length = min(200, max_length)
        adjusted_min_length = min(80, min_length)
    elif text_word_count > 200:
        adjusted_max_length = min(150, max_length)
        adjusted_min_length = min(60, min_length)
    else:
        adjusted_max_length = min(120, max_length)
        adjusted_min_length = min(50, min_length)

    # Generate summary with mixed precision for faster inference
    with torch.no_grad(), autocast(enabled=torch.cuda.is_available()):
        summary_ids = model.generate(
            **inputs,
            max_length=adjusted_max_length,
            min_length=adjusted_min_length,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True,
            no_repeat_ngram_size=3
        )

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    # Strictly enforce summary relevance and factuality
    if not is_summary_relevant(summary, text) or len(summary.split('.')) < 2:
        summary = extractive_summary(text)

    # Ensure summary ends with a period and is capitalized
    if summary and not summary.endswith('.'):
        summary = summary + '.'
    if summary and not summary[0].isupper():
        summary = summary[0].upper() + summary[1:]
    return summary

# Helper to check if summary is relevant to text
def is_summary_relevant(summary, text):
    summary_words = set(summary.lower().split())
    text_words = set(text.lower().split())
    # If less than 40% of summary words are in text, or summary is too short, it's not relevant
    overlap = len(summary_words & text_words) / (len(summary_words) + 1e-6)
    return overlap > 0.4 and len(summary_words) > 8

# Simple extractive fallback: first sentence + most informative sentence
def extractive_summary(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if not sentences:
        return ""
    # Always select at least two sentences if possible
    selected = []
    for s in sentences:
        if len(s.split()) > 8:
            selected.append(s.strip())
        if len(selected) == 2:
            break
    if not selected:
        return sentences[0].strip()
    return ' '.join(selected)

def generate_summary_batch(texts, model=None, tokenizer=None, max_length=150, min_length=50):
    """
    Generate abstractive summaries for a batch of texts with optimized performance
    
    Args:
        texts (list of str): List of texts to summarize
        model: The pre-loaded summarization model (optional)
        tokenizer: The pre-loaded tokenizer (optional)
        max_length (int): Maximum length of the summary (in tokens)
        min_length (int): Minimum length of the summary (in tokens)
        
    Returns:
        list of str: List of summaries
    """
    # Load model if not provided
    if model is None or tokenizer is None:
        model, tokenizer = load_model()
    
    # Get device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Skip empty or very short texts
    valid_texts = []
    skip_indices = []
    for i, text in enumerate(texts):
        if not text or len(text.split()) < 10:
            skip_indices.append(i)
        else:
            valid_texts.append(text)
    
    # If all texts are invalid, return empty summaries
    if not valid_texts:
        return [""] * len(texts)
    
    # Preprocess text to remove unnecessary elements - use list comprehension for efficiency
    valid_texts = [preprocess_text(text) for text in valid_texts]
    
    # Calculate dynamic summary lengths based on text lengths
    text_lengths = [len(text.split()) for text in valid_texts]
    avg_length = sum(text_lengths) // len(text_lengths) if text_lengths else 0
    dynamic_max_length, dynamic_min_length = calculate_target_length(avg_length)
    
    # Use the provided lengths if they're smaller than the calculated ones
    final_max_length = min(dynamic_max_length, max_length)
    final_min_length = min(dynamic_min_length, min_length)
    
    # Tokenize batch with efficient padding and truncation
    inputs = tokenizer(valid_texts, return_tensors="pt", padding="longest", truncation=True, max_length=tokenizer.model_max_length)
    
    # Move all inputs to GPU if available and free memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate summaries with torch.no_grad() and autocast for memory efficiency and speed
    with torch.no_grad(), autocast(enabled=torch.cuda.is_available()):
        summary_ids = model.generate(
            **inputs,
            max_length=final_max_length,
            min_length=final_min_length,
            num_beams=4,  
            length_penalty=1.0,
            early_stopping=True,
            no_repeat_ngram_size=3
        )
    
    # Decode batch
    summaries = tokenizer.batch_decode(summary_ids, skip_special_tokens=True)
    
    # Free memory after generation
    del inputs, summary_ids
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Post-process summaries
    return process_summaries(summaries, valid_texts, texts, skip_indices)

def process_summaries(summaries, valid_texts, original_texts, skip_indices):
    """
    Process generated summaries and reinsert them into the original text order
    
    Args:
        summaries (list): List of generated summaries for valid texts
        valid_texts (list): List of valid texts that were summarized
        original_texts (list): Original list of texts including skipped ones
        skip_indices (list): Indices of texts that were skipped
        
    Returns:
        list: List of summaries in the original order
    """
    # Initialize result list with empty strings
    result = [""] * len(original_texts)
    
    # Process each summary
    valid_idx = 0
    for i in range(len(original_texts)):
        if i in skip_indices:
            # Skip indices get empty summaries
            continue
        else:
            if valid_idx < len(summaries):
                summary = summaries[valid_idx]
                
                # Ensure summary is properly formatted
                if summary:
                    # Ensure summary ends with a period
                    if not summary.endswith('.'):
                        summary = summary + '.'
                    # Ensure summary starts with a capital letter
                    if not summary[0].isupper():
                        summary = summary[0].upper() + summary[1:]
                
                result[i] = summary
                valid_idx += 1
    
    return result

def preprocess_text(text):
    """
    Preprocess text to remove unnecessary elements
    
    Args:
        text (str): The text to preprocess
        
    Returns:
        str: Preprocessed text
    """
    if not text:
        return ""
    
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def calculate_target_length(text_length):
    """
    Calculate appropriate summary length based on input text length
    
    Args:
        text_length (int): Length of input text in words
        
    Returns:
        tuple: (max_length, min_length) in tokens
    """
    if text_length > 1000:
        return 150, 80  # ~112 words max, ~60 words min
    elif text_length > 500:
        return 110, 70  # ~80 words max, ~50 words min
    else:
        return 90, 50   # ~60 words max, ~30 words min

def adaptive_batch_processing(texts, model, tokenizer, initial_batch_size=8, progress_callback=None):
    """
    Process texts with adaptive batch size to handle memory constraints
    
    Args:
        texts (list): List of texts to summarize
        model: The summarization model
        tokenizer: The tokenizer
        initial_batch_size (int): Starting batch size
        progress_callback: Optional callback function to update progress
        
    Returns:
        list: List of summaries
    """
    # Use consistent batch size of 8 for better performance and progress tracking
    batch_size = initial_batch_size
    
    # Calculate appropriate summary lengths based on average text length
    avg_length = sum(len(text.split()) for text in texts) // len(texts) if texts else 0
    max_length, min_length = calculate_target_length(avg_length)
    
    # Clean up GPU memory before processing
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Process texts in smaller batches with tqdm progress tracking
    all_summaries = []
    total_texts = len(texts)
    processed = 0
    
    # Create progress bar with consistent format
    pbar = tqdm(total=total_texts, desc="Summarizing articles", unit="article", bar_format=tqdm_format)
    
    # Process in batches of batch_size
    while processed < total_texts:
        end_idx = min(processed + batch_size, total_texts)
        current_batch = texts[processed:end_idx]
        
        try:
            batch_summaries = generate_summary_batch(current_batch, model, tokenizer, max_length, min_length)
            all_summaries.extend(batch_summaries)
            
            # Update progress
            batch_size_processed = end_idx - processed
            processed += batch_size_processed
            
            # Update progress bar
            pbar.update(batch_size_processed)
            pbar.set_postfix({"batch_size": batch_size})
            
            # Call progress callback if provided
            if progress_callback:
                progress_callback(batch_size_processed)
                
        except RuntimeError as e:  # CUDA OOM error
            if 'CUDA out of memory' in str(e):
                batch_size = max(1, batch_size // 2)
                torch.cuda.empty_cache()
                pbar.set_postfix({"batch_size": batch_size, "status": "reduced due to OOM"})
                
                if batch_size < 1:
                    # Force CPU processing as last resort
                    pbar.set_postfix({"status": "CPU fallback"})
                    device_backup = next(model.parameters()).device
                    model = model.to('cpu')
                    
                    # Process remaining texts on CPU
                    remaining_batch = texts[processed:]
                    remaining_summaries = generate_summary_batch(remaining_batch, model, tokenizer, max_length, min_length)
                    all_summaries.extend(remaining_summaries)
                    
                    # Update progress for remaining
                    pbar.update(len(remaining_batch))
                    if progress_callback:
                        progress_callback(len(remaining_batch))
                        
                    model = model.to(device_backup)  # Restore original device
                    break
            else:
                # Re-raise if it's not a memory error
                raise
    
    # Close progress bar
    pbar.close()
    
    return all_summaries

def format_key_points(key_points):
    """
    Format key points to ensure they are complete sentences with proper punctuation
    Args:
        key_points (list of str): List of key points
    Returns:
        list of str: Formatted key points
    """
    formatted = []
    for kp in key_points:
        kp = kp.strip()
        if not kp:
            continue
        # Remove any key point that is not present in the article (factual enforcement)
        # This assumes the original article text is available as context; if not, skip this check
        # For now, just ensure it's a complete sentence
        if not kp.endswith('.'):
            kp += '.'
        if kp and not kp[0].isupper():
            kp = kp[0].upper() + kp[1:]
        formatted.append(kp)
    # Remove duplicates and empty points
    unique = []
    seen = set()
    for kp in formatted:
        if kp not in seen and len(kp.split()) > 4:
            unique.append(kp)
            seen.add(kp)
    return unique

def is_unrelated_keypoint(point):
    # Filter out generic or unrelated keypoints
    unrelated_phrases = [
        'met gala', 'kiara advani', 'category:', 'finance', 'fashionable looks',
        'check out', 'continued', 'ahead of', 'category', 'business', 'entertainment', 'sports', 'technology'
    ]
    for phrase in unrelated_phrases:
        if phrase in point.lower():
            return True
    return False

def generate_article_summary(article_text):
    """
    Generate an abstractive summary for a single article text
    Args:
        article_text (str): The text to summarize
    Returns:
        str: The generated summary
    """
    # Load model if not already loaded
    model, tokenizer = load_model()
    
    if not article_text or len(article_text.strip()) < 100:
        return ""  # Skip too short content

    # Preprocess text to remove unnecessary elements
    article_text = preprocess_text(article_text)
    
    # Calculate appropriate summary length based on input text length
    text_word_count = len(article_text.split())
    max_length, min_length = calculate_target_length(text_word_count)

    # Prepare inputs with efficient padding and truncation
    inputs = tokenizer(article_text, truncation=True, padding="longest", max_length=512, return_tensors="pt")
    
    # Move inputs to GPU if available and ensure memory is optimized
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    # Generate summary with optimized parameters
    with torch.no_grad(), autocast(enabled=torch.cuda.is_available()):
        summary_ids = model.generate(
            inputs["input_ids"],
            num_beams=4,  # Reduced from 5 to 4 for better memory efficiency
            min_length=min_length,
            max_length=max_length,
            length_penalty=1.0,
            early_stopping=True,
            no_repeat_ngram_size=3
        )

    # Decode
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    # Free memory after generation
    del inputs, summary_ids
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Strictly enforce summary relevance and factuality
    if not is_summary_relevant(summary, article_text) or len(summary.split('.')) < 2:
        summary = extractive_summary(article_text)

    # Ensure summary ends with a period and is capitalized
    if summary and not summary.endswith('.'):
        summary = summary + '.'
    if summary and not summary[0].isupper():
        summary = summary[0].upper() + summary[1:]
        
    return summary.strip()

def summarize_articles(articles, batch_size=8):
    """
    Generate summaries for a list of articles using optimized batching
    
    Args:
        articles (list): List of article dictionaries
        batch_size (int): Number of articles to process in each batch
        
    Returns:
        list: List of articles with added 'summary' field
    """
    # Create a progress bar for the overall process
    main_pbar = tqdm(total=100, desc="Summarization pipeline", unit="%", bar_format=tqdm_format)
    main_pbar.update(10)  # Initial progress for setup
    
    # Load model once for all articles with half precision for faster inference
    model, tokenizer = load_model()
    main_pbar.update(20)  # Update progress after model loading
    
    # Clean up GPU memory before processing
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Process all articles with consistent parameters
    summarized_articles = []
    
    # Prepare content to summarize
    contents = []
    article_indices = []
    
    # Create a progress bar for content preparation
    # prep_pbar = tqdm(total=len(articles), desc="Preparing articles", unit="article", bar_format=tqdm_format)
    
    for i, article in enumerate(articles):
        content = article.get("content", "")
        
        # Only take articles with some content (reduced threshold to ensure processing)
        if content and len(content) >= 50:
            contents.append(content)
            article_indices.append(i)
        else:
            # Process articles without content immediately
            article_copy = article.copy()
            article_copy["summary"] = "No content available for summarization."
            summarized_articles.append(article_copy)
        
        # Update preparation progress bar
        prep_pbar.update(1)
    
    # Close preparation progress bar
    prep_pbar.close()
    main_pbar.update(10)  # Update main progress after preparation
    
    # Performance monitoring
    total_start_time = time.time()
    
    # Process articles in batches using adaptive_batch_processing
    if contents:
        # Process all contents at once using adaptive batch processing
        # The adaptive_batch_processing function now has its own progress bar
        summaries = adaptive_batch_processing(contents, model, tokenizer, initial_batch_size=batch_size)
        
        main_pbar.update(40)  # Update main progress after batch processing
        
        # Create a progress bar for adding summaries back to articles
        summary_pbar = tqdm(total=len(summaries), desc="Finalizing summaries", unit="article", bar_format=tqdm_format)
        
        # Add summaries to articles
        for i, summary in enumerate(summaries):
            if summary:
                article_idx = article_indices[i]
                article_copy = articles[article_idx].copy()
                article_copy["summary"] = summary
                summarized_articles.append(article_copy)
            
            # Update summary progress bar
            summary_pbar.update(1)
        
        # Close summary progress bar
        summary_pbar.close()
    
    # Calculate total time
    total_time = time.time() - total_start_time
    
    # Update final progress
    main_pbar.update(20)
    main_pbar.set_postfix({"time": f"{total_time:.2f}s", "articles": len(summarized_articles)})
    main_pbar.close()
    
    return summarized_articles

if __name__ == "__main__":
    # Verify CUDA is properly configured
    print("\n===== CUDA Configuration Test =====")
    is_cuda_available = verify_cuda_usage()
    print("===================================\n")
    
    # Test the summarizer with batch processing
    test_texts = [
        """
        Apple has announced a new iPhone model with improved camera capabilities and longer battery life. 
        The iPhone 13 Pro features a redesigned camera system with three new lenses, including a telephoto 
        lens with 3x optical zoom. The new A15 Bionic chip provides up to 50% faster performance than the 
        leading competition. Apple claims the battery can last up to 22 hours of video playback. 
        The phone will be available in four colors and starts at $999 for the base model.
        """,
        """
        NASA's Perseverance rover has successfully collected its first rock sample on Mars. 
        The sample, about the size of a piece of chalk, was drilled from a Martian rock nicknamed "Rochette." 
        This historic achievement marks the first time a spacecraft has collected a rock sample from another planet 
        with the intent to return it to Earth. Scientists hope the samples will provide insights into Mars' 
        climate and geology, potentially revealing if the planet once harbored microbial life.
        """
    ]
    
    # Load model once for all tests
    model, tokenizer = load_model()
    
    # Test batch processing with GPU acceleration
    print("\n===== Testing Batch Processing =====")
    start_mem = torch.cuda.memory_allocated(0) / 1024**2 if is_cuda_available else 0
    
    batch_summaries = generate_summary_batch(test_texts, model, tokenizer)
    
    if is_cuda_available:
        end_mem = torch.cuda.memory_allocated(0) / 1024**2
        print(f"Memory used for batch processing: {end_mem - start_mem:.2f}MB")
    
    for i, (text, summary) in enumerate(zip(test_texts, batch_summaries)):
        print(f"\nTest {i+1}:")
        print(f"Original text length: {len(text.split())} words")
        print(f"Summary length: {len(summary.split())} words")
        print(f"Summary: {summary}")
    
    # Test with category-specific parameters
    print("\n===== Testing Category-Specific Parameters =====")
    tech_params = {"max_length": 100, "min_length": 30}
    science_params = {"max_length": 110, "min_length": 40}
    
    # Create a mixed batch with different parameters
    if is_cuda_available:
        torch.cuda.empty_cache()  # Clear GPU memory between tests
        print(f"GPU memory after cache clear: {torch.cuda.memory_allocated(0) / 1024**2:.2f}MB")
    
    mixed_summaries = [
        generate_summary_batch([test_texts[0]], model, tokenizer, max_length=tech_params["max_length"], min_length=tech_params["min_length"])[0],
        generate_summary_batch([test_texts[1]], model, tokenizer, max_length=science_params["max_length"], min_length=science_params["min_length"])[0]
    ]
    
    print("\nCategory-specific summaries:")
    print(f"Tech summary: {mixed_summaries[0]}")
    print(f"Science summary: {mixed_summaries[1]}")
    
    # Final memory cleanup
    if is_cuda_available:
        torch.cuda.empty_cache()
        print(f"\nFinal GPU memory usage: {torch.cuda.memory_allocated(0) / 1024**2:.2f}MB / {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f}GB")
        print("CUDA acceleration successfully verified!")
    else:
        print("\nRunning on CPU - consider setting up CUDA for faster processing.")

# Define the model path
model_path = "path/to/model"  # Update this to your model's path

# Check if the model file exists
# if os.path.exists(model_path):
#     summarizer = pipeline("summarization", model=model_path, framework="pt")
# else:
#     st.error("Model file not found!")

# Example usage of the summarizer
def summarize_article(article):
    summary = summarizer(article)
    return summary
