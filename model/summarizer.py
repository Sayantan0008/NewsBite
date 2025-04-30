from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch
from torch.cuda.amp import autocast
from tqdm import tqdm
import re
import warnings
import os
import tensorflow as tf
import time
import concurrent.futures

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=all, 1=info, 2=warning, 3=error
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Display PyTorch version and CUDA information
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    # Clean up GPU memory at start
    torch.cuda.empty_cache()

# Global variables to avoid reloading model
MODEL = None
TOKENIZER = None

def verify_cuda_usage():
    """
    Verify that CUDA is being properly used and display GPU memory usage
    """
    if torch.cuda.is_available():
        # Create a small test tensor and verify it's on CUDA
        test_tensor = torch.tensor([1.0, 2.0, 3.0])
        test_tensor = test_tensor.to("cuda")
        print(f"Test tensor device: {test_tensor.device}")
        
        # Display GPU memory usage
        allocated = torch.cuda.memory_allocated(0) / 1024**2
        reserved = torch.cuda.memory_reserved(0) / 1024**2
        print(f"GPU Memory: {allocated:.2f}MB allocated, {reserved:.2f}MB reserved")
        return True
    else:
        print("CUDA is not available. Using CPU for inference (slower).")
        return False

def load_model():
    """
    Load the Pegasus summarization model and tokenizer with half-precision for faster inference
    """
    global MODEL, TOKENIZER
    
    if MODEL is None or TOKENIZER is None:
        print("Loading Pegasus summarization model...")
        model_name = "google/pegasus-xsum"
        
        # Load tokenizer and model
        TOKENIZER = PegasusTokenizer.from_pretrained(model_name)
        MODEL = PegasusForConditionalGeneration.from_pretrained(model_name)
        
        # Move to GPU if available with half precision for faster inference and less VRAM usage
        if torch.cuda.is_available():
            MODEL = MODEL.half()  # Convert to half precision (FP16)
            MODEL = MODEL.to("cuda")
            print(f"Model moved to GPU with half precision: {next(MODEL.parameters()).device}")
            verify_cuda_usage()
        else:
            print("Using CPU for model inference (slower)")
    
    return MODEL, TOKENIZER

def generate_summary(text, model, tokenizer, max_length=150, min_length=70):
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
        adjusted_min_length = min(40, min_length)

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

def generate_summary_batch(texts, model=None, tokenizer=None, max_length=150, min_length=60):
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
    
    # Preprocess text to remove unnecessary elements
    valid_texts = [preprocess_text(text) for text in valid_texts]
    
    # Tokenize batch
    inputs = tokenizer(valid_texts, return_tensors="pt", padding=True, truncation=True, max_length=tokenizer.model_max_length)
    
    # Move all inputs to GPU if available
    if torch.cuda.is_available():
        inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate summaries with torch.no_grad() and autocast for memory efficiency and speed
    with torch.no_grad(), autocast(enabled=torch.cuda.is_available()):
        summary_ids = model.generate(
            **inputs,
            max_length=max_length,
            min_length=min_length,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True,
            no_repeat_ngram_size=3
        )
    
    # Decode batch
    summaries = tokenizer.batch_decode(summary_ids, skip_special_tokens=True)
    
    # Post-process summaries (implement inline instead of calling missing function)
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

def adaptive_batch_processing(texts, model, tokenizer, initial_batch_size=8):
    """
    Process texts with adaptive batch size to handle memory constraints
    
    Args:
        texts (list): List of texts to summarize
        model: The summarization model
        tokenizer: The tokenizer
        initial_batch_size (int): Starting batch size
        
    Returns:
        list: List of summaries
    """
    # Start with a larger batch size if GPU is available
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        if gpu_mem > 8:
            initial_batch_size = 20
        elif gpu_mem > 6:
            initial_batch_size = 16
        elif gpu_mem > 4:
            initial_batch_size = 12
        else:
            initial_batch_size = 8
    
    batch_size = initial_batch_size
    print(f"Starting with batch size: {batch_size}")
    
    # Calculate appropriate summary lengths based on average text length
    avg_length = sum(len(text.split()) for text in texts) // len(texts) if texts else 0
    max_length, min_length = calculate_target_length(avg_length)
    
    # Clean up GPU memory before processing
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    while batch_size >= 1:
        try:
            return generate_summary_batch(texts, model, tokenizer, max_length, min_length)
        except RuntimeError as e:  # CUDA OOM error
            if 'CUDA out of memory' in str(e):
                batch_size = batch_size // 2
                torch.cuda.empty_cache()
                print(f"Reducing batch size to {batch_size} due to memory constraints")
                if batch_size < 1:
                    # Force CPU processing as last resort
                    print("Warning: Processing on CPU as GPU memory is insufficient")
                    device_backup = next(model.parameters()).device
                    model = model.to('cpu')
                    results = generate_summary_batch(texts, model, tokenizer, max_length, min_length)
                    model = model.to(device_backup)  # Restore original device
                    return results
            else:
                # Re-raise if it's not a memory error
                raise
    
    # Fallback if all else fails
    return [text[:200] + "..." if text else "" for text in texts]

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

def summarize_articles(articles, batch_size=8):
    """
    Generate summaries for a list of articles using optimized batching
    
    Args:
        articles (list): List of article dictionaries
        batch_size (int): Number of articles to process in each batch
        
    Returns:
        list: List of articles with added 'summary' field
    """
    print(f"🔵 Starting summarization for {len(articles)} articles...")
    
    # Load model once for all articles with half precision for faster inference
    model, tokenizer = load_model()
    
    # Clean up GPU memory before processing
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Optimize batch size based on GPU availability
    if torch.cuda.is_available():
        # If we have a GPU, we can use a larger batch size with half precision
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        print(f"Total GPU memory: {gpu_mem:.2f}GB")
        
        # Adjust batch size based on available GPU memory - larger batches with half precision
        if gpu_mem > 8:
            batch_size = 16  # Can use larger batch with half precision
        elif gpu_mem > 6:
            batch_size = 12
        elif gpu_mem > 4:
            batch_size = 8
        else:
            batch_size = 6
            
        print(f"Using optimized batch size: {batch_size} for GPU processing with half precision")
    
    # Process all articles with consistent parameters
    summarized_articles = []
    
    # Track generated summaries to prevent duplicates
    generated_summaries = set()
    
    # Prepare content to summarize
    contents = []
    article_indices = []
    
    for i, article in enumerate(articles):
        content = article.get("content", "")
        
        if content and len(content.split()) >= 10:
            contents.append(content)
            article_indices.append(i)
        else:
            # Process articles without content immediately
            article_copy = article.copy()
            article_copy["summary"] = "No content available for summarization."
            summarized_articles.append(article_copy)
    
    # Performance monitoring
    total_start_time = time.time()
    
    # Sort articles by length for better GPU utilization
    sorted_indices = sorted(range(len(contents)), key=lambda i: len(contents[i].split()))
    
    # Create batches for processing
    batches = [sorted_indices[i:i+batch_size] for i in range(0, len(sorted_indices), batch_size)]
    
    # Create progress bar with position and leave parameters
    progress_bar = tqdm(total=len(batches), desc="Summarizing articles", position=0, leave=True)
    
    # Process in fixed size batches with similar-length articles
    for batch_indices in batches:
        batch_texts = [contents[idx] for idx in batch_indices]
        
        # Time each batch
        batch_start_time = time.time()
        
        # Calculate appropriate summary lengths based on average text length
        avg_length = sum(len(text.split()) for text in batch_texts) // len(batch_texts) if batch_texts else 0
        max_length, min_length = calculate_target_length(avg_length)
        min_length = min(min_length, 40)  # Reduced minimum length requirement for speed
        
        # Generate summaries with mixed precision
        batch_summaries = generate_summary_batch(
            batch_texts, model, tokenizer, 
            max_length=max_length,
            min_length=min_length
        )
        
        # Calculate timing information
        batch_time = time.time() - batch_start_time
        avg_time_per_article = batch_time / len(batch_texts)
        
        # Update tqdm description with timing information instead of printing separately
        progress_bar.set_description(f"Summarizing articles (avg {avg_time_per_article:.2f}s/article)")
        progress_bar.update(1)
        
        # Process results directly without individual article retry logic
        for j, summary in enumerate(batch_summaries):
            if j < len(batch_indices):  # Safety check
                article_idx = article_indices[batch_indices[j]]
                article_copy = articles[article_idx].copy()
                content = article_copy.get("content", "")
                title = article_copy.get("title", "")
                
                # Handle NaN or empty summaries
                if not summary or summary.lower() == "nan" or summary in generated_summaries:
                    # If summary is a duplicate or invalid, generate a new one based on content
                    if content and len(content.split()) >= 20:
                        # Extract key sentences from the article
                        sentences = re.split(r'(?<=[.!?])\s+', content)
                        if len(sentences) >= 3:
                            # Use first sentence and an important middle sentence
                            first_sent = sentences[0].strip()
                            middle_sent = sentences[min(len(sentences) // 2, len(sentences) - 1)].strip()
                            
                            # Combine with title if available
                            if title and len(title) > 10:
                                summary = f"{title}. {first_sent} {middle_sent}"
                            else:
                                summary = f"{first_sent} {middle_sent}"
                        else:
                            # Use what we have if not enough sentences
                            summary = ". ".join([s.strip() for s in sentences if s.strip()])
                    elif title and len(title) > 10:
                        # Use title as fallback
                        summary = title
                    else:
                        summary = "No summary available for this article."
                
                # Ensure summary is substantial (at least 2-3 lines)
                if summary and len(summary.split()) < 15 and content and len(content.split()) > 50:
                    # Extract key facts from content
                    sentences = re.split(r'(?<=[.!?])\s+', content)
                    key_sentences = []
                    
                    # Get first sentence
                    if sentences and len(sentences) > 0:
                        key_sentences.append(sentences[0].strip())
                    
                    # Get a sentence from the middle that contains numbers or key entities
                    for i in range(1, min(len(sentences), 10)):
                        if re.search(r'\d+|\$|percent|million|billion|trillion', sentences[i].lower()):
                            key_sentences.append(sentences[i].strip())
                            break
                    
                    # Get another important sentence if needed
                    if len(key_sentences) < 2 and len(sentences) > 2:
                        for s in sentences[1:min(len(sentences), 5)]:
                            if len(s.split()) > 8 and s not in key_sentences:
                                key_sentences.append(s.strip())
                                break
                    
                    # Combine sentences into a better summary
                    if key_sentences:
                        summary = " ".join(key_sentences)
                
                # Ensure summary ends with a period and is a complete sentence
                if summary and not summary.endswith('.'):
                    summary = summary + '.'
                if summary and not summary[0].isupper():
                    summary = summary[0].upper() + summary[1:]
                
                # Check for duplicate summaries and ensure uniqueness
                if summary in generated_summaries:
                    # Add article-specific information to make it unique
                    if title and len(title) > 5:
                        unique_prefix = title.split()[0:2]
                        summary = f"{' '.join(unique_prefix)}: {summary}"
                
                # Add to tracking set to prevent future duplicates
                generated_summaries.add(summary)
                article_copy["summary"] = summary
                
                # Extract key points for better context
                if "key_points" not in article_copy or not article_copy["key_points"]:
                    if content and len(content.split()) >= 50:  # Only for longer articles
                        # Extract important sentences as key points
                        sentences = re.split(r'(?<=[.!?])\s+', content)
                        key_points = []
                        
                        # First sentence is usually important
                        if sentences and len(sentences) > 0:
                            key_points.append(sentences[0])
                        
                        # Look for sentences with numbers, dates, or key entities
                        for s in sentences[1:min(len(sentences), 15)]:
                            if re.search(r'\d+|\$|percent|million|billion|trillion|yesterday|today|tomorrow', s.lower()):
                                if s not in key_points:
                                    key_points.append(s)
                                    if len(key_points) >= 3:
                                        break
                        
                        # If we don't have enough key points, add another important sentence
                        if len(key_points) < 2 and len(sentences) > 5:
                            for s in sentences[1:min(len(sentences), 10)]:
                                if len(s.split()) > 10 and s not in key_points:
                                    key_points.append(s)
                                    break
                        
                        article_copy["key_points"] = "\n".join(key_points)
                
                # Format key points if they exist
                if "key_points" in article_copy and article_copy["key_points"]:
                    article_copy["key_points"] = format_key_points(article_copy["key_points"])
                
                summarized_articles.append(article_copy)
    
    # Close the progress bar
    progress_bar.close()
    print("✅ Summarization completed.")
    
    
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