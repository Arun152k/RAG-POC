# RAG POC

This repository contains the code for a Retrieval-Augmented Generation (RAG) application. The application uses Streamlit to provide an interactive interface where users can ask questions based on indexed documents stored in AWS S3. The system uses embeddings from HuggingFace and OpenAI to generate responses.

## Introduction to RAG Pipeline

Retrieval-Augmented Generation (RAG) is a technique that combines information retrieval and natural language generation. It first retrieves relevant documents or text passages from a knowledge base and then uses a language model to generate answers based on the retrieved content. This approach ensures that the generated answers are accurate and contextually relevant.

In this application, we use a merger retriever to combine results from multiple retrievers, an embeddings redundant filter to remove redundant documents, and a long context reorderer to optimize the context for the language model. These components are integrated into a document compressor pipeline, which provides contextually compressed retrievals, ensuring high-quality and relevant responses to user queries.

## Features

- Load PDF documents from AWS S3.
- Index documents using Chroma vector store.
- Use HuggingFace and OpenAI embeddings.
- Retrieve and answer questions based on the indexed documents.
- Interactive interface using Streamlit.

## Prerequisites

- Python 3.9 or higher
- AWS account with access to S3
- OpenAI API key
- Docker (optional, for containerized deployment)

## Installation

1. Clone the repository:

    ```sh
    git clone https://github.com/yourusername/rag-poc.git
    cd rag-poc
    ```

2. Create a virtual environment and activate it:

    ```sh
    python3 -m venv venv
    source venv/bin/activate
    ```

3. Install the dependencies:

    ```sh
    pip install -r requirements.txt
    ```

4. Set up your environment variables in a `.env` file:

    ```env
    OPENAI_API_KEY=your_openai_api_key
    AWS_ACCESS_KEY_ID=your_aws_access_key_id
    AWS_SECRET_ACCESS_KEY=your_aws_secret_access_key
    REGION_NAME=your_aws_region
    ```

## Usage

1. Start the Streamlit application:

    ```sh
    streamlit run app.py
    ```

2. Open your browser and navigate to `http://localhost:8501`.

3. Ask questions based on the indexed documents.

## Docker Deployment

To deploy the application using Docker:

1. Build the Docker image:

    ```sh
    docker build -t rag-poc .
    ```

2. Run the Docker container:

    ```sh
    docker run -p 8501:8501 --env-file .env rag-poc
    ```

3. Open your browser and navigate to `http://localhost:8501`.

## Code Overview

### Main Components

- **Loading PDF documents from S3**: The function `load_pdfs_from_s3` fetches PDF documents from an S3 bucket and processes them using `PyPDF2`.
- **Embedding Setup**: The function `embedding` sets up embeddings using HuggingFace and OpenAI models.
- **Vector Store Setup**: The function `setup_vector_stores` and `add_to_store` manage the creation and indexing of document vectors using Chroma.
- **Retriever and QA Setup**: Functions `setup_retrievers`, `setup_RAG_pipeline_retriever`, and `setup_LLM_QA` configure the retriever and QA pipeline.
- **Streamlit Interface**: The `main` function sets up the Streamlit interface, handles user input, and displays the chat history.

