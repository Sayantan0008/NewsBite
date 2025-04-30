# NewsBite Storage Optimization

This document outlines the storage optimization techniques implemented in the NewsBite application to reduce disk space usage while maintaining application performance.

## Implemented Optimizations

### 1. Data Compression

- **Gzip Compression**: All news data files are automatically compressed using gzip, reducing file size by approximately 70-80%.
- **Selective Compression**: The most recent file remains uncompressed for faster access, while older files are automatically compressed.
- **Transparent Access**: The application can seamlessly read both compressed and uncompressed files without user intervention.

### 2. Content Optimization

- **Content Truncation**: Article content is optimized during the initial saving process, limiting excessively long articles to a reasonable length (5000 characters by default).
- **On-Demand Loading**: In the dashboard, article content is initially truncated to 1000 characters, with a "Show full content" button to load the complete text only when needed.

### 3. Pagination

- **Limited Loading**: The dashboard implements pagination to load only a subset of articles at once, reducing memory usage.
- **Configurable Page Size**: Users can adjust the number of articles displayed per page (5-50) to balance between performance and convenience.

### 4. Data Retention Policy

- **Automatic Cleanup**: Files older than 30 days are automatically removed to prevent unlimited storage growth.
- **Configurable Retention**: The retention period can be adjusted in the `cleanup_old_files` function parameter.

### 5. Memory Optimization

- **LRU Caching**: The data loading function is cached using Python's `lru_cache` decorator to prevent redundant file operations.

## Implementation Details

### Key Components

- **utils/data_storage.py**: Contains utilities for compressed data storage and content optimization.
- **app/dashboard.py**: Implements pagination, content truncation, and file maintenance.
- **main.py**: Uses the optimized storage utilities during the pipeline execution.

### Usage

The optimizations are applied automatically when running the application. No additional configuration is required.

```python
# Example: Running the pipeline with optimized storage
python main.py

# Example: Viewing the dashboard with optimized loading
streamlit run app/dashboard.py
```

## Performance Impact

- **Storage Reduction**: Expect 70-90% reduction in disk space usage compared to the previous implementation.
- **Memory Usage**: Significantly reduced memory footprint when browsing large news collections.
- **Load Time**: Slight increase in initial load time for compressed files, offset by reduced file sizes and pagination.

## Categorization Optimization

- **Enhanced Keyword System**: The expanded subcategory system with comprehensive keyword sets improves categorization accuracy while maintaining efficient storage.
- **Context-Aware Analysis**: Improved keyword matching with context awareness reduces false positives without increasing storage requirements.
- **Subcategory Storage**: Subcategories are stored alongside main categories, adding minimal storage overhead while providing significant additional information.
