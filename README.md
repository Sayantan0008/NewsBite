# 🧠 NewsBite: Full ML News Pipeline

![Python Version](https://img.shields.io/badge/python-3.12%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0%2B-red)
![CUDA](https://img.shields.io/badge/CUDA-11.8%2B-yellow)

NewsBite is an AI-powered news aggregation and analysis system that fetches the latest news articles, generates concise summaries, categorizes them, and presents them in an interactive dashboard.

## 📋 Features

- **News Fetching**: Retrieves the latest headlines and content from NewsAPI.org
- **Summarization**: Uses Google's Pegasus model with GPU optimizations, including half-precision, autocast, and optimized batch processing to generate consistent summaries for articles.
- **Categorization**: Classifies articles into categories using BERT
- **Data Storage**: Saves processed articles in compressed CSV format with optimization
- **Interactive Dashboard**: Streamlit-based UI with filtering and search capabilities
- **PyTorch/Streamlit Compatibility**: Enhanced patching system to prevent runtime errors
- **GPU Acceleration**: Optimized for CUDA with configurable batch sizes for faster processing

## 🗂️ Project Structure

```
newsbite/
├── data/                 # Compressed CSV files with processed news articles
├── app/                  # Streamlit dashboard
├── docs/                 # Documentation files
│   └── storage_optimization.md  # Storage optimization details
├── model/                # Pegasus & BERT models
├── services/             # NewsAPI fetcher
├── utils/                # Preprocessing, helper tools
│   ├── data_storage.py   # Optimized storage utilities
│   ├── torch_streamlit_patch.py  # PyTorch compatibility patch
│   └── ...               # Other utility modules
├── main.py               # End-to-end runner
├── run_dashboard.py      # Dashboard launcher
├── streamlit_patch.py    # Enhanced PyTorch/Streamlit compatibility patch
├── requirements.txt      # Project dependencies
└── README.md             # This file
```

## ⚡ Performance Optimization

NewsBite is optimized for high performance with GPU acceleration:

- **Half-precision (FP16)**: Uses 16-bit floating point for faster computation and memory efficiency
- **Adaptive batch sizes**: Dynamically adjusts batch sizes based on GPU capability
- **Memory optimization**: Configurable to utilize up to 2GB of GPU memory
- **CUDA autocast**: Automatic mixed precision for optimal inference speed

Batch size configuration:
| GPU Memory | Batch Size | Notes |
|------------|------------|-------|
| > 8GB | 16 | High-end GPUs (RTX 3080, RTX 4070, RTX 4080 etc.) |
| > 6GB | 12 | Mid-range GPUs (RTX 3060, RTX 4060 etc.) |
| > 4GB | 8 | Budget GPUs (GTX 1660, etc.) |
| < 4GB | 4 | Entry-level GPUs |

## 🚀 Getting Started

### Prerequisites

- Python 3.12 or compatible version
- NewsAPI.org API key (get one at [newsapi.org](https://newsapi.org))
- CUDA 11.8 or newer (for GPU acceleration, recommended)

### Installation Requirements

All required dependencies are listed in the `requirements.txt` file. The main requirements include:

- **Core**: numpy, pandas, requests
- **ML & NLP**: transformers, torch, datasets, sentencepiece, regex
- **News API**: newsapi-python
- **UI**: streamlit
- **Utilities**: tqdm, python-dotenv, matplotlib, scikit-learn

### GPU Acceleration

NewsBite uses PyTorch for ML models and can benefit significantly from GPU acceleration:

1. **Check CUDA compatibility**: The latest PyTorch versions work with CUDA 11.8 or newer
2. **Install CUDA Toolkit**: Download from [NVIDIA CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive)
3. **Install GPU-enabled PyTorch**:
   ```
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```
4. **Verify installation**: Run this command to check if PyTorch can access your GPU:
   ```python
   python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')"
   ```

The summarizer and categorizer components will automatically use GPU acceleration if available.

### Installation

1. Clone the repository:

   ```
   git clone https://github.com/Sayantan0008/NewsBite.git
   cd newsbite
   ```

2. Install dependencies using the setup script (recommended):

   ```
   python setup.py
   ```

   This script will install all required dependencies including Streamlit.

   Alternatively, you can install dependencies manually:

   ```
   pip install -r requirements.txt
   ```

   If you encounter any issues with Streamlit installation, you can install it separately:

   ```
   pip install streamlit>=1.0.0,<2.0.0
   ```

   To verify Streamlit is properly installed:

   ```
   python -c "import streamlit; print('Streamlit version:', streamlit.__version__)"
   ```

3. Create a `.env` file in the project root with your NewsAPI key:

   ```
   NEWS_API_KEY=your_api_key_here
   ```

   Note: The `.env` file is already created for you. Just replace `your_api_key_here` with your actual NewsAPI.org API key.

### Running the Pipeline

1. Run the complete pipeline to fetch, summarize, and categorize news:

   ```
   python main.py
   ```

2. Launch the dashboard to view the processed articles:

   ```
   python run_dashboard.py
   ```

   This launcher automatically applies the PyTorch/Streamlit compatibility patch to prevent runtime errors.

### Storage Optimization

NewsBite implements several storage optimization techniques:

- Automatic gzip compression of older data files
- Content truncation for excessively long articles
- Pagination in the dashboard for efficient memory usage
- Configurable data retention policy

See `docs/storage_optimization.md` for detailed information.

## 🔧 Components

### News Fetcher

The `services/news_fetcher.py` module connects to NewsAPI.org to retrieve the latest news articles. It extracts relevant fields like title, content, source, URL, and publication date.

### Summarizer

The `model/summarizer.py` module uses the Pegasus model (google/pegasus-xsum) to generate abstractive summaries of news articles. It processes each article to create concise 1-3 line summaries.

Key features:

- **Adaptive summary length**: Automatically adjusts summary length based on article content
- **Half-precision (FP16)**: Reduces memory usage and increases speed on GPUs
- **Batch processing**: Processes multiple articles simultaneously for higher throughput
- **Extractive fallback**: Falls back to extractive summarization if abstractive summarization fails
- **Empty content handling**: Gracefully handles missing or minimal article content
- **CUDA optimization**: Utilizes GPU memory efficiently with autocast and memory management

### Categorizer

The `model/categorizer.py` module uses a BERT-based model to classify articles into predefined categories like Politics, Tech, Sports, etc. It implements a hybrid approach combining keyword-based analysis and machine learning predictions. The system includes:

- **Main Category Classification**: Classifies articles into 8 main categories (Politics, Tech, Sports, Health, Entertainment, World, Business, Science)
- **Subcategory Classification**: Further categorizes articles into specific subcategories using an enhanced keyword-based system
- **Context-Aware Analysis**: Improved keyword matching with context awareness for better accuracy
- **Expanded Vocabulary**: Comprehensive keyword sets for each category and subcategory

### Preprocessing

The `utils/preprocessing.py` module cleans and prepares the news articles for processing by the ML models. It handles text normalization, content extraction, and filtering.

### Dashboard

The `app/dashboard.py` module provides a Streamlit-based user interface for viewing the processed news articles. It includes filters for category, source, and date, as well as a search function.

### PyTorch/Streamlit Compatibility Patch

The `streamlit_patch.py` and `utils/torch_streamlit_patch.py` modules provide a comprehensive solution to prevent runtime errors caused by interactions between PyTorch's custom class system and Streamlit's file watcher. The patch intercepts problematic attribute access patterns and provides safe alternatives, ensuring smooth operation of the dashboard with PyTorch models.

### Data Storage Optimization

The `utils/data_storage.py` module implements storage optimization techniques including automatic compression, content truncation, and efficient data loading. These optimizations reduce disk space usage while maintaining application performance.

## 🔍 Troubleshooting

### Common Issues

#### Streamlit Not Found

If you encounter a "No module named streamlit" error when running the dashboard:

1. Ensure you've installed all dependencies correctly:

   ```
   pip install -r requirements.txt
   ```

2. Try installing Streamlit directly:

   ```
   pip install streamlit>=1.0.0,<2.0.0
   ```

3. Verify your virtual environment is activated (if using one)

4. Check if Streamlit is installed in your Python environment:
   ```
   python -c "import pkg_resources; print(pkg_resources.get_distribution('streamlit').version)"
   ```

#### PyTorch/Streamlit Compatibility Issues

If you see errors related to PyTorch and Streamlit compatibility:

1. Ensure you're using the provided `run_dashboard.py` script which applies necessary patches
2. Update to the latest versions of both libraries
3. If issues persist, try running Streamlit directly: `streamlit run app/dashboard.py`

#### GPU Memory Issues

If you encounter GPU memory errors:

1. Reduce batch size in `model/summarizer.py` by adjusting the `batch_size` parameter
2. Close other GPU-intensive applications
3. Try running with CPU-only mode by setting `CUDA_VISIBLE_DEVICES=''` before running the script

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- [NewsAPI.org](https://newsapi.org) for providing the news data API
- [Hugging Face](https://huggingface.co) for the Transformers library and pre-trained models
- [Streamlit](https://streamlit.io) for the dashboard framework
