import torch
from model.summarizer import load_model, generate_summary
import time

# Check CUDA availability
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    # Clean up GPU memory at start
    torch.cuda.empty_cache()

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

# Load model
print("\nLoading model...")
start_time = time.time()
model, tokenizer = load_model()
load_time = time.time() - start_time
print(f"Model loaded in {load_time:.2f} seconds")

# Check GPU memory usage
if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated(0) / 1024**2
    reserved = torch.cuda.memory_reserved(0) / 1024**2
    print(f"GPU Memory: {allocated:.2f}MB allocated, {reserved:.2f}MB reserved")

# Generate summary
print("\nGenerating summary...")
start_time = time.time()
summary = generate_summary(test_text, model, tokenizer)
summarize_time = time.time() - start_time
print(f"Summary generated in {summarize_time:.2f} seconds")

# Print summary
print("\nSummary:")
print(summary)

# Check GPU memory usage after summarization
if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated(0) / 1024**2
    reserved = torch.cuda.memory_reserved(0) / 1024**2
    print(f"\nGPU Memory after summarization: {allocated:.2f}MB allocated, {reserved:.2f}MB reserved")