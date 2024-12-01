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
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
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



def retrieve_context_by_document(query, retriever, uploaded_files, top_k_per_doc=10):
    document_context = {}
    for uploaded_file in uploaded_files:
        retrieved_chunks = retriever.get_relevant_documents(
            query,
            search_type="similarity",
            top_k=top_k_per_doc,
            metadata_filters={"source_document": uploaded_file.name}
        )
        document_context[uploaded_file.name] = [
            f"{chunk.metadata['chunk_id']}: {chunk.page_content}" for chunk in retrieved_chunks
        ]
    formatted_context = "\n\n".join(
        f"[Source: {doc_name}]\n{''.join(chunks)}" for doc_name, chunks in document_context.items()
    )
    return document_context, formatted_context


def create_worker_chain(worker_id, context, task):
    """
    Creates a worker chain for an assigned task with the specific context provided.

    Args:
        worker_id (int): The ID of the Worker LLM.
        context (str): The context provided for this Worker to analyze.
        task (str): The specific high-level task assigned to the Worker.

    Returns:
        LLMChain: The configured LLMChain for the Worker.
    """
    worker_prompt = PromptTemplate(
        input_variables=["context", "question", "worker_id", "task"],
        template=(
            "Worker {worker_id}: Your role is to complete the assigned task using the provided context. "
            "Your assigned task is:\n{task}\n\n"
            "Analyze the context to extract concise and specific facts, insights, or relevant details "
            "that directly contribute to completing the task. Use only the provided context and focus strictly "
            "on your assigned task. Cite references (chunk IDs) where applicable. "
            "Limit your response to 3 sentences.\n\n"
            "Provided Context:\n{context}\n\n"
            "User Query:\n{question}\n\n"
            "Findings (Facts and Insights Only):"
        )
    )
    worker_llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.1-8b-instant")
    return LLMChain(llm=worker_llm, prompt=worker_prompt)





def director_command(query, retriever, uploaded_files):
    director_llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.1-8b-instant")

    # Retrieve context grouped by document
    document_context, formatted_context = retrieve_context_by_document(query, retriever, uploaded_files)

    all_source_docs = "\n".join([uploaded_file.name for uploaded_file in uploaded_files])

    # High-level prompt for the Director
    prompt = (
        f"You are the Director LLM. Your task is to define up to 3 high-level steps necessary to address the user's query. "
        f"Focus on general and broad actions that Workers can interpret and act upon without micromanaging details. "
        f"These steps should ensure comprehensive coverage of the query using the uploaded documents.\n\n"
        f"User Query: {query}\n\n"
        f"Uploaded Source Documents:\n{all_source_docs}\n\n"
        "Define the steps:\n"
        "- The steps should outline general objectives for analyzing the documents.\n"
        "- Avoid providing detailed substeps or assigning tasks to individual Workers.\n"
        "- Ensure that the steps cover different aspects of the query comprehensively.\n\n"
        "Output the steps as a numbered list of general actions."
    )

    director_command_text = director_llm.invoke([("user", prompt)]).content.strip()

    return director_command_text, formatted_context







def coordinator_summary(worker_responses):
    coordinator_llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.1-8b-instant")
    combined_responses = "\n\n".join([f"Worker {i + 1}:\n{response}" for i, response in enumerate(worker_responses)])
    
    prompt = (
        "You are the Coordinator LLM. Your role is to synthesize the responses from Workers into a single, accurate, "
        "and concise answer to the user's query. Ensure the final response integrates insights from all Workers, "
        "addresses the query directly, and cites relevant sources. Avoid assumptions or extraneous information.\n\n"
        f"Insights from Workers:\n{combined_responses}\n\n"
        "Final Answer:"
    )
    
    return coordinator_llm.invoke([("user", prompt)]).content.strip()





# Evaluator Feedback
def evaluator_feedback(query, summary, retriever, uploaded_files):
    evaluator_llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.1-8b-instant")
    _, formatted_context = retrieve_context_by_document(query, retriever, uploaded_files)

    validation_prompt = (
        f"User Query: {query}\n\n"
        f"Proposed Answer by Coordinator:\n{summary}\n\n"
        f"Relevant Context from Database:\n{formatted_context}\n\n"
        "Evaluate the accuracy and relevancy of the proposed answer. Respond with:\n"
        "- Approve if the answer is accurate, relevant, and addresses the query.\n"
        "- Disapprove and provide actionable feedback if the answer is inaccurate or incomplete.\n"
        "Highlight discrepancies, missing information, or errors in the response."
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
        col1, col2, col3, col4 = st.columns([2, 2, 2, 2])

        # Director LLM: Scopes the task and assigns subtasks
        with col1:
            st.header("Director LLM")
            try:
                retriever = st.session_state.vectors.as_retriever()
                director_command_text, formatted_context = director_command(query, retriever, uploaded_files)
                st.write(f"**Director Command:**\n\n{director_command_text}")
            except Exception as e:
                st.error(f"Error processing Director LLM: {e}")
                logging.error(f"Director LLM error: {e}")
                break


        
        # Worker LLMs: Perform assigned tasks
        with col2:
            st.header("Worker LLMs")
            worker_responses = []
            try:
                # Parse the Director's output into high-level tasks
                director_steps = director_command_text.split("\n")
                tasks = [step.strip() for step in director_steps if step.strip()]  # Filter out empty lines

                # Ensure we have enough steps for Workers
                if len(tasks) < 3:
                    st.warning("Director did not provide sufficient steps for Workers. Restarting process.")
                    break

                # Assign tasks to Workers and process context
                for worker_id, task in enumerate(tasks, start=1):
                    worker_chain = create_worker_chain(worker_id, formatted_context, task)
                    worker_input = {
                        "context": formatted_context,
                        "question": query,
                        "worker_id": worker_id,
                        "task": task
                    }
                    worker_response = worker_chain.run(worker_input)
                    worker_responses.append(worker_response)
                    st.markdown(f"**Worker {worker_id} Task:** {task}\n\n**Response:**\n{worker_response}", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error processing Worker LLMs: {e}")
                logging.error(f"Worker LLM error: {e}")
                break


        
        # Coordinator LLM: Synthesizes Worker responses
        with col3:
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
        with col4:
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






