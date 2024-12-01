# Worker LLM looks at different parts of the vector database

import os
import numpy as np
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv
import streamlit as st
import tempfile

# Load environment variables
load_dotenv()
API_KEYS = os.getenv("GROQ_API_KEYS").split(",")  # Comma-separated API keys in .env file

# Sidebar for PDF Upload and Processing
with st.sidebar:
    st.title("Upload and Process PDFs")
    uploaded_files = st.file_uploader("Upload PDF files", accept_multiple_files=True, type=["pdf"])

    if uploaded_files and "vectors" not in st.session_state:
        st.info("Processing documents and building vector database...")

        # Initialize session state variables
        if "final_documents" not in st.session_state:
            st.session_state.final_documents = []
        if "embeddings" not in st.session_state:
            st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        # Step 1: Load and Split Documents
        for uploaded_file in uploaded_files:
            # Save uploaded file to a temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.read())
                temp_file_path = temp_file.name

            # Load the temporary file using PyPDFLoader
            loader = PyPDFLoader(temp_file_path)
            documents = loader.load()

            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(documents)
            st.session_state.final_documents.extend(chunks)

        st.success("PDFs processed and documents split into chunks.")

        # Step 2: Process documents in batches
        batch_size = 100
        all_embeddings = []

        for i in range(0, len(st.session_state.final_documents), batch_size):
            batch = st.session_state.final_documents[i:i + batch_size]
            texts = [doc.page_content for doc in batch]
            try:
                batch_embeddings = st.session_state.embeddings.embed_documents(texts)
                all_embeddings.extend(batch_embeddings)
                st.sidebar.write(f"Processed {min(i + batch_size, len(st.session_state.final_documents))} out of {len(st.session_state.final_documents)} documents")
            except Exception as e:
                st.sidebar.error(f"Error processing batch {i // batch_size + 1}: {str(e)}")
                raise

        # Step 3: Create FAISS index
        embeddings_array = np.array(all_embeddings, dtype=np.float32)
        texts = [doc.page_content for doc in st.session_state.final_documents]
        metadatas = [doc.metadata for doc in st.session_state.final_documents]

        st.session_state.vectors = FAISS.from_embeddings(
            text_embeddings=list(zip(texts, embeddings_array)),
            embedding=st.session_state.embeddings,
            metadatas=metadatas
        )

        st.success("Vector database built successfully.")

# Function to create Conversational Retrieval Chain
def create_worker_chain(retriever, memory):
    """
    Creates a ConversationalRetrievalChain for worker LLMs.
    """
    # Combine docs chain (uses QA logic)
    combine_docs_chain = load_qa_chain(
        llm=ChatGroq(groq_api_key=API_KEYS[0], model_name="llama3-8b-8192"),
        chain_type="stuff"  # Combine retrieved docs into an answer
    )

    # Question generator chain
    question_gen_prompt = PromptTemplate(
        input_variables=["chat_history", "question"],
        template="Given the conversation history:\n\n{chat_history}\n\nRefine this question:\n\n{question}"
    )
    question_generator = LLMChain(
        llm=ChatGroq(groq_api_key=API_KEYS[0], model_name="llama3-8b-8192"),
        prompt=question_gen_prompt
    )

    # Create ConversationalRetrievalChain
    return ConversationalRetrievalChain(
        retriever=retriever,
        memory=memory,
        combine_docs_chain=combine_docs_chain,
        question_generator=question_generator
    )

# Main Panel for Chat Interface
st.title("Agentic RAG Chatbot")
query = st.text_input("Ask a question based on the uploaded documents:")

if query:
    # st.info("Processing your query through multiple worker LLMs...")
    
    # Columns for displaying output
    col1, col2 = st.columns(2)

    with col1:
        st.header("Worker Outputs")
    with col2:
        st.header("Final Summarized Response")

    # Shard FAISS for worker diversity
    def shard_vector_db(vector_db, num_workers):
        all_docs = list(vector_db.docstore._dict.values())  # Retrieve all documents
        num_docs = len(all_docs)
        shard_size = num_docs // num_workers
        shards = []

        for i in range(num_workers):
            start_idx = i * shard_size
            end_idx = num_docs if i == num_workers - 1 else (i + 1) * shard_size
            shard_docs = all_docs[start_idx:end_idx]  # Slice the documents
            shard_texts = [doc.page_content for doc in shard_docs]
            shard_metadatas = [doc.metadata for doc in shard_docs]
            shard_embeddings = vector_db.embedding_function.embed_documents(shard_texts)
            shard_db = FAISS.from_embeddings(
                text_embeddings=list(zip(shard_texts, shard_embeddings)),
                embedding=vector_db.embedding_function,
                metadatas=shard_metadatas
            )
            shards.append(shard_db)

        return shards


    sharded_vector_dbs = shard_vector_db(st.session_state.vectors, len(API_KEYS))
    
    # Coordinator LLM summarization
    coordinator_llm = ChatGroq(groq_api_key=API_KEYS[0], model_name="llama3-8b-8192")

    def summarize_responses(responses):
        """
        Summarize worker responses using the coordinator LLM.
        """
        combined_worker_responses = (
            "Here are the responses from the workers:\n"
            + "\n".join(f"Worker {i + 1}: {response}" for i, response in enumerate(responses))
            + "\n\nSummarize these responses into a single coherent answer."
        )

        # Define the messages in the expected structured format
        messages = [
            (
                "system", 
                (
                    "You are a coordinator LLM responsible for summarizing worker outputs. "
                    "Your goal is to provide the most accurate and coherent answer to the user's question based on the workers' responses. "
                    "When some workers provide explicit answers and others do not find the answer, prioritize the explicit answers but acknowledge that others may not have found the information. "
                    "If responses contradict each other, weigh responses that are clear, complete, and plausible more heavily. "
                    "Avoid repeating vague or incomplete responses. "
                    "Your final response should be concise, clear, and confidently present the most likely correct answer."
                )
            ),
            ("user", combined_worker_responses)
        ]
        # Pass the structured messages to the coordinator LLM
        return coordinator_llm.invoke(messages)


    # Main execution
    if query:
        st.info("Processing your query through multiple worker LLMs...")
        
        # Worker LLMs generate diverse responses
        worker_responses = []
        for i, (api_key, shard_db) in enumerate(zip(API_KEYS, sharded_vector_dbs)):
            retriever = shard_db.as_retriever()
            memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
            worker_chain = create_worker_chain(retriever, memory)
            response = worker_chain.run({"question": query})
            worker_responses.append(response)

            with col1:
                st.write(f"Worker {i + 1} Output:")
                st.write(response)

        st.success("Workers have completed their tasks.")

        # Display the Final Summarized Response in Chat Format
        try:
            final_response = summarize_responses(worker_responses)
            
            
            final_response_text = final_response.content.strip()
            
            with col2:
                st.markdown("### Final Summarized Response")
                st.markdown(f"**User:** {query}")  # Display the user's query
                st.markdown(f"**Bot:** {final_response_text}")  # Display the bot's summarized response
        except Exception as e:
            st.error(f"Error summarizing responses: {e}")




