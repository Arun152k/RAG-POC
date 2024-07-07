import pysqlite3
import sys

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import os
import openai
import io
import streamlit as st
from dotenv import load_dotenv
import boto3
from PyPDF2 import PdfReader
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.schema import Document
from langchain_community.embeddings import (
    HuggingFaceEmbeddings,
    OpenAIEmbeddings,
    HuggingFaceBgeEmbeddings,
)
from langchain.retrievers.merger_retriever import MergerRetriever
from langchain.document_transformers import EmbeddingsRedundantFilter
from langchain.chains import RetrievalQA
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.document_transformers import LongContextReorder
import logging
from streamlit_chromadb_connection.chromadb_connection import ChromadbConnection

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


def get_openai_api_key():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OpenAI API key not found. Please set it in the .env file.")
        raise ValueError("OpenAI API key not found. Please set it in the .env file.")
    return api_key


def get_bucket_name():
    return "rag-poc-1"


def aws_client():
    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    region_name = os.getenv("REGION_NAME")

    s3_client = None
    try:
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name,
        )
        logger.info("AWS client created successfully")
    except Exception as e:
        logger.error(f"AWS client creation unsuccessful: {e}")

    return s3_client


def load_pdfs_from_s3(s3_client):
    bucket_name = get_bucket_name()
    response = s3_client.list_objects_v2(Bucket=bucket_name)
    documents = {"Resume": [], "Essay": []}
    for obj in response.get("Contents", []):
        if obj["Key"].endswith(".pdf"):
            pdf_obj = s3_client.get_object(Bucket=bucket_name, Key=obj["Key"])
            pdf_content = pdf_obj["Body"].read()
            pdf = PdfReader(io.BytesIO(pdf_content))
            text = ""
            for page_num in range(len(pdf.pages)):
                text += pdf.pages[page_num].extract_text()
            doc = Document(page_content=text, metadata={"source": obj["Key"]})
            if "Resume" in obj["Key"]:
                documents["Resume"].append(doc)
            elif "Essay" in obj["Key"]:
                documents["Essay"].append(doc)
    return documents


def embedding():
    logger.info("Setting up embeddings.")
    hf_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": False},
    )

    hf_bge_embeddings = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-large-en",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": False},
    )

    openai_embeddings = OpenAIEmbeddings(openai_api_key=get_openai_api_key())

    return hf_embeddings, hf_bge_embeddings, openai_embeddings


def setup_embeddings():
    embeddings = {}
    embeddings["all-MiniLM-L6-v2"], embeddings["bge-large-en"], embeddings["openAI"] = (
        embedding()
    )
    return embeddings


def setup_vector_stores(documents, embeddings):
    logger.info("Setting up Vector Stores.")
    vector_stores = {}
    i = 0
    for collection_name, document in documents.items():
        if i % 2 == 0:
            embedding_choice = embeddings["all-MiniLM-L6-v2"]
        else:
            embedding_choice = embeddings["bge-large-en"]
        i += 1
        vector_stores[collection_name] = add_to_store(
            document, embedding_choice, collection_name
        )
    return vector_stores


def add_to_store(document, embedding, collection_name):
    logger.info(f"Adding documents to store: {collection_name}")

    configuration = {"client": "PersistentClient", "path": "/tmp/.chroma"}

    conn = st.connection(
        name="persistent_chromadb", type=ChromadbConnection, **configuration
    )

    embedding_function_name = "DefaultEmbeddingFunction"
    conn.create_collection(
        collection_name=collection_name,
        embedding_function_name=embedding_function_name,
        embedding_config={},
        metadata={"hnsw:space": "cosine"},
    )

    vectorstore = Chroma.from_documents(
        documents=document,
        embedding=embedding,
        client=conn,
        collection_name=collection_name,
    )

    return vectorstore


def setup_retrievers(vector_stores):
    retrievers = {}
    for vector_store in vector_stores:
        retrievers[vector_store] = vector_stores[vector_store].as_retriever(
            search_type="mmr", search_kwargs={"k": 5, "include_metadata": True}
        )
    return retrievers


def setup_RAG_pipeline_retriever(embeddings, retrievers):
    lotr = MergerRetriever(retrievers=list(retrievers.values()))

    filter = EmbeddingsRedundantFilter(embeddings=embeddings["openAI"])
    reordering = LongContextReorder()
    pipeline = DocumentCompressorPipeline(transformers=[filter, reordering])
    RAGPipelineRetriever = ContextualCompressionRetriever(
        base_compressor=pipeline,
        base_retriever=lotr,
        search_kwargs={"k": 5, "include_metadata": True},
    )
    return RAGPipelineRetriever


def setup_custom_prompt():
    custom_prompt = """
    You are an expert assistant. Use the following context to answer the question. If you do not know the answer, ask the user to ask the question later. 
        
    Context:
    {context}

    Question: {question}

    Answer:
    """
    return custom_prompt


def setup_LLM_QA(RAGPipelineRetriever):
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0.9)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=RAGPipelineRetriever,
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": PromptTemplate(
                template=setup_custom_prompt(),
                input_variables=["context", "question"],
            )
        },
    )
    return qa


def display_chat_history():
    for chat in st.session_state.chat_history:
        st.write(f"**Query:** {chat['query']}")
        st.write("**Response:**")
        output = chat["response"]
        lines = output.split("\n")
        formatted_output = ""

        for line in lines:
            if line.startswith("1.") or line.startswith("2.") or line.startswith("3."):
                formatted_output += f"\n{line}\n"
            else:
                formatted_output += f"    {line.strip()}\n"

        st.write(formatted_output)
        st.write("---")


def main():

    st.set_page_config(
        page_title="RAG POC",
        layout="centered",
        initial_sidebar_state="auto",
    )

    st.title("RAG POC")
    st.header(
        "Ask a question based on the [documents](https://drive.google.com/drive/folders/1ljXhVwdBulNN0aU4nAq6pBw8ScfDSsZs?usp=drive_link) indexed:"
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "qa" not in st.session_state:
        s3_client = aws_client()
        documents = load_pdfs_from_s3(s3_client)
        embeddings = setup_embeddings()
        vector_stores = setup_vector_stores(documents, embeddings)
        retrievers = setup_retrievers(vector_stores)
        RAGPipelineRetriever = setup_RAG_pipeline_retriever(embeddings, retrievers)
        st.session_state.qa = setup_LLM_QA(RAGPipelineRetriever)
        st.session_state.documents_loaded = True
        logger.info("Initialization complete.")

    question = st.text_input("Your Query:", "")

    if st.button("Submit"):
        if question:
            try:
                qa = st.session_state.qa
                result = qa(question)
                output = result["result"]
                st.session_state.chat_history.append(
                    {"query": question, "response": output}
                )
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")

    display_chat_history()


if __name__ == "__main__":
    main()
