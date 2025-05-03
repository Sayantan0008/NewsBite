import torch
from NewsBite.model.summarizer import load_model, generate_article_summary
import time
from tqdm import tqdm

# Initialize progress bar for setup
with tqdm(total=100, desc="Initializing", unit="%", bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
    # Check CUDA availability silently
    pbar.update(30)
    if torch.cuda.is_available():
        # Clean up GPU memory at start
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    pbar.update(70)

# Test text
test_text = """
Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to intelligence displayed by animals and humans. 
Example tasks in which this is done include speech recognition, computer vision, translation between (natural) languages, as well as other mappings of inputs.
AI applications include advanced web search engines (e.g., Google), recommendation systems (used by YouTube, Amazon, and Netflix), 
understanding human speech (such as Siri and Alexa), self-driving cars (e.g., Waymo), generative or creative tools (ChatGPT and AI art), 
automated decision-making, and competing at the highest level in strategic game systems (such as chess and Go).
As machines become increasingly capable, tasks considered to require "intelligence" are often removed from the definition of AI, 
a phenomenon known as the AI effect. For instance, optical character recognition is frequently excluded from things considered to be AI, 
having become a routine technology.
"""

# Load model with progress tracking
with tqdm(total=100, desc="Loading model", unit="%", bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
    start_time = time.time()
    model, tokenizer = load_model()
    load_time = time.time() - start_time
    pbar.update(100)

# Display memory and timing info in a clean way
info = []
info.append(f"Model loaded in {load_time:.2f} seconds")

# Check GPU memory usage
if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated(0) / 1024**2
    reserved = torch.cuda.memory_reserved(0) / 1024**2
    info.append(f"GPU Memory: {allocated:.2f}MB allocated, {reserved:.2f}MB reserved")

# Display collected info
for line in info:
    tqdm.write(line)

# Generate summary with progress tracking
with tqdm(total=100, desc="Generating summary", unit="%", bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
    start_time = time.time()
    pbar.update(10)  # Show initial progress
    summary = generate_article_summary(test_text)
    pbar.update(90)  # Complete the progress
    summarize_time = time.time() - start_time

# Display summary and timing
tqdm.write(f"\nSummary generated in {summarize_time:.2f} seconds")
tqdm.write("\nSummary:")
tqdm.write(summary)

# Check GPU memory usage after summarization
if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated(0) / 1024**2
    reserved = torch.cuda.memory_reserved(0) / 1024**2
    tqdm.write(f"\nGPU Memory after summarization: {allocated:.2f}MB allocated, {reserved:.2f}MB reserved")