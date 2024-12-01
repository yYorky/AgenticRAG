# need to figure out why sometimes worker do not have access to some documents, even though it is chunked. linked to context size

import os
import numpy as np
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import streamlit as st
import tempfile
import logging

st.set_page_config(layout="wide")
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # Single API key

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set or loaded. Please check your .env file or environment variables.")

# Sidebar for User Input and PDF Upload
with st.sidebar:
    st.title("Agentic RAG Chatbot")
    uploaded_files = st.file_uploader("Upload PDF files", accept_multiple_files=True, type=["pdf"])
    query = st.text_input("Ask a question based on the uploaded documents:")

    if uploaded_files and "vectors" not in st.session_state:
        st.info("Processing documents and building vector database...")

        if "final_documents" not in st.session_state:
            st.session_state.final_documents = []
        if "embeddings" not in st.session_state:
            st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        
        # Updated Chunking Process
        
        # Updated Chunking Process
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.read())
                temp_file_path = temp_file.name

            loader = PyPDFLoader(temp_file_path)
            documents = loader.load()

            # Process each document independently to ensure boundaries are maintained
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            document_chunks = text_splitter.split_documents(documents)

            for i, chunk in enumerate(document_chunks):
                # Add metadata to ensure each chunk is tied only to its source document
                chunk.metadata["chunk_id"] = f"{uploaded_file.name}_chunk_{i + 1}"
                chunk.metadata["source_document"] = uploaded_file.name

            st.session_state.final_documents.extend(document_chunks)


        
        # Validate that all chunks belong to their respective source document
        for chunk in st.session_state.final_documents:
            assert chunk.metadata["source_document"] in chunk.metadata["chunk_id"], (
                f"Chunk {chunk.metadata['chunk_id']} does not match its source document: {chunk.metadata['source_document']}"
            )


        # Generate embeddings and create FAISS index
        batch_size = 100
        all_embeddings = []
        for i in range(0, len(st.session_state.final_documents), batch_size):
            batch = st.session_state.final_documents[i:i + batch_size]
            texts = [doc.page_content for doc in batch]
            batch_embeddings = st.session_state.embeddings.embed_documents(texts)
            all_embeddings.extend(batch_embeddings)

        embeddings_array = np.array(all_embeddings, dtype=np.float32)
        texts = [doc.page_content for doc in st.session_state.final_documents]
        metadatas = [doc.metadata for doc in st.session_state.final_documents]
        st.session_state.vectors = FAISS.from_embeddings(
            text_embeddings=list(zip(texts, embeddings_array)),
            embedding=st.session_state.embeddings,
            metadatas=metadatas
        )
        st.success("Vector database built successfully.")



def retrieve_context_by_document(query, retriever, uploaded_files, top_k_per_doc=5):
    """
    Retrieve a fixed number of chunks (top_k_per_doc) from each document to ensure equal representation.
    
    Args:
        query (str): The query string.
        retriever: The vector retriever object.
        uploaded_files (list): List of uploaded file objects.
        top_k_per_doc (int): Number of chunks to retrieve per document.

    Returns:
        document_context (dict): Mapping of document names to retrieved chunks.
        formatted_context (str): Concatenated string of all chunks for worker input.
    """
    document_context = {}
    for uploaded_file in uploaded_files:
        # Retrieve chunks for the current document
        retrieved_chunks = retriever.get_relevant_documents(
            query,
            search_type="similarity",
            top_k=top_k_per_doc,
            metadata_filters={"source_document": uploaded_file.name}
        )

        # Fallback if fewer than top_k_per_doc chunks are available
        if len(retrieved_chunks) < top_k_per_doc:
            logging.warning(
                f"Document '{uploaded_file.name}' has only {len(retrieved_chunks)} relevant chunks. Expected {top_k_per_doc}."
            )

        document_context[uploaded_file.name] = [
            f"{chunk.metadata['chunk_id']}: {chunk.page_content}" for chunk in retrieved_chunks
        ]

    # Combine chunks into a formatted string for workers
    formatted_context = "\n\n".join(
        f"[Source: {doc_name}]\n{''.join(chunks)}" for doc_name, chunks in document_context.items()
    )
    return document_context, formatted_context



def worker_task(query, formatted_context):
    """
    Worker LLM: Tries to answer the user query directly using the provided context.
    """
    worker_llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama3-70b-8192")
    
    # Simplified prompt for Worker
    prompt = (
        f"Use the provided context to answer the user's query as accurately as possible.\n\n"
        f"User Query: {query}\n\n"
        f"Relevant Context:\n{formatted_context}\n\n"
        "Answer (max 3 sentences):"
    )
    
    response = worker_llm.invoke([("user", prompt)]).content.strip()
    return response


def coordinator_summary(worker_responses):
    """
    Coordinator LLM: Synthesizes Worker responses into a final summary.
    """
    coordinator_llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama3-70b-8192")
    combined_responses = "\n\n".join([f"Worker {i + 1}:\n{response}" for i, response in enumerate(worker_responses)])
    
    # Simplified prompt for Coordinator
    prompt = (
        "Summarize the responses from Workers into one clear and accurate answer to the user's query.\n\n"
        f"Worker Responses:\n{combined_responses}\n\n"
        "Final Answer:"
    )
    
    return coordinator_llm.invoke([("user", prompt)]).content.strip()


def evaluator_feedback(query, summary, retriever, uploaded_files):
    """
    Evaluator LLM: Validates the final summary against the query and context.
    """
    evaluator_llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama3-70b-8192")
    _, formatted_context = retrieve_context_by_document(query, retriever, uploaded_files)

    # Simplified prompt for Evaluator
    validation_prompt = (
        f"Review the proposed answer against the user's query and the provided context.\n\n"
        f"User Query: {query}\n\n"
        f"Proposed Answer: {summary}\n\n"
        f"Relevant Context:\n{formatted_context}\n\n"
        "Reply with 'Approve' if the answer is correct, or provide feedback if changes are needed."
    )

    evaluator_response = evaluator_llm.invoke([("user", validation_prompt)]).content.strip()
    if "approve" in evaluator_response.lower():
        return "approve", evaluator_response
    else:
        return "disapprove", evaluator_response


# Main Execution
if query and uploaded_files:
    st.title("Agentic RAG Chatbot Results")
    iteration = 0
    max_iterations = 3  # Avoid infinite loop

    while iteration < max_iterations:
        iteration += 1
        st.subheader(f"Iteration {iteration}")
        col1, col2, col3 = st.columns([3, 3, 3])

        # Worker LLM: Directly attempts to answer the query
        with col1:
            st.header("Worker LLMs")
            worker_responses = []
            try:
                retriever = st.session_state.vectors.as_retriever()
                _, formatted_context = retrieve_context_by_document(query, retriever, uploaded_files)

                # Workers provide responses
                for worker_id in range(3):  # Simulate 3 workers for redundancy
                    worker_response = worker_task(query, formatted_context)
                    worker_responses.append(worker_response)
                    st.markdown(f"**Worker {worker_id + 1} Response:**\n\n{worker_response}", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error processing Worker LLMs: {e}")
                logging.error(f"Worker LLM error: {e}")
                break

        # Coordinator LLM: Synthesizes Worker responses
        with col2:
            st.header("Coordinator LLM")
            try:
                if worker_responses:
                    final_summary = coordinator_summary(worker_responses)
                    st.markdown(f"**Final Summary:**\n\n{final_summary}", unsafe_allow_html=True)
                else:
                    st.warning("No responses from Worker LLMs to summarize.")
                    final_summary = None
            except Exception as e:
                st.error(f"Error synthesizing responses: {e}")
                logging.error(f"Coordinator LLM error: {e}")
                break

        # Evaluator LLM: Validates and fact-checks the final summary
        with col3:
            st.header("Evaluator LLM")
            if final_summary:
                try:
                    approval_status, feedback = evaluator_feedback(query, final_summary, retriever, uploaded_files)
                    if approval_status == "approve":
                        st.success("Final response approved by Evaluator LLM!")
                        st.markdown(f"**Final Output:**\n\n{final_summary}", unsafe_allow_html=True)
                        break
                    else:
                        st.warning("Evaluator provided feedback for improvement.")
                        st.markdown(f"**Feedback:**\n\n{feedback}", unsafe_allow_html=True)
                        query = feedback  # Restart with feedback as the refined query
                except Exception as e:
                    st.error(f"Error during evaluation: {e}")
                    logging.error(f"Evaluator LLM error: {e}")
                    break

    # Handle termination after maximum iterations
    if iteration >= max_iterations:
        st.error("Maximum iterations reached. Process terminated.")
else:
    st.warning("Please upload documents and provide a query to proceed.")







