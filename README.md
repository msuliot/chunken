# CHUNKEN - CHUNK Extraction Node

CHUNKEN is designed to further process the text extracted by TEXTEN by chunking it into manageable parts and creating embeddings for these chunks. It integrates with OpenAI for generating embeddings and uses Pinecone and MongoDB for storage and retrieval of chunked data.

Key Features
- Text Preprocessing: Cleans and preprocesses text, removing unnecessary elements like headers, footers, and URLs.
- Chunking: Divides large text files into smaller chunks for easier processing and analysis.
- Embeddings Generation: Uses OpenAI embeddings to generate vector representations of text chunks.
- Storage Integration: Stores chunk metadata and embeddings in MongoDB and Pinecone for efficient retrieval.
- Orphaned Chunk Management: Identifies and deletes orphaned chunks in Pinecone to maintain data integrity.

## Git Repositories
- https://github.com/msuliot/texten.git
- https://github.com/msuliot/webtexten.git
- https://github.com/msuliot/chunken.git
- https://github.com/msuliot/datamyn.git

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)

## Prerequisites

Before you begin, ensure you have met the following requirements:
- You have installed Python 3.7 or later.
- You have a working internet connection.

## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/msuliot/chunken.git
    cd texten
    ```

2. **Set up a virtual environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate   # On Windows, use `venv\Scripts\activate`
    ```

3. **Install the dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run the CHUNKEN application, use the following command:

```bash
python app.py
```

### Configuration

The configuration is managed through a `config.json` file. Create a configuration file with the following structure:

```json
{
    "input_directories": [
      "path/to/text/output/directory"
    ],
    "database": "blades-of-grass-demo",
    "namespace": "demo24",
    "chunk_size": 1800, 
    "chuck_extension_limit": 248,
    "scheduler_interval": 60
}
```