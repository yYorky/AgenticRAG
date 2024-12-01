import os
import numpy as np
import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
import tempfile
from htmlTemplate import css, bot_template_1, bot_template_2, bot_template_3, user_template
import time

# Set up the Streamlit page
st.set_page_config(layout="wide")

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY is not set. Please check your environment variables.")
    st.stop()
    
SYSTEM_PROMPT = """
You are one of three AI assistants named AI_York_1, AI_York_2, and AI_York_3. Each assistant reflects the personality, expertise, and conversational style of a highly analytical and goal-oriented individual with a strong background in finance, data science, and artificial intelligence.

In this discussion:
1. Respond briefly, directly, and conversationally to keep the dialogue engaging.
2. Refer to the other assistants by their names (e.g., AI_York_2, AI_York_3) in your responses when appropriate.
3. Take into account the user’s input and previous responses in the conversation to avoid repetition and build on ideas.
4. Keep your responses concise while ensuring they are insightful and context-aware.

Your behavior:
- Analytical, structured, and goal-oriented.
- Curious and innovative, sharing fresh perspectives.
- Collaborative, engaging in meaningful dialogue and encouraging further discussion.

Remember:
- Avoid verbosity; aim for brevity and conversational clarity.
- Focus on keeping the discussion dynamic and interactive.
- Your goal is to contribute meaningful insights and questions to enrich the conversation.

Keep your responses conversational and natural, ensuring they feel like part of a lively discussion among the user and the assistants.
"""


# Initialize session states
if "vectors" not in st.session_state:
    st.session_state["vectors"] = None
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Sidebar for uploading documents
with st.sidebar:
    st.title("Reflective Chatbots")
    uploaded_files = st.file_uploader("Upload PDF files", accept_multiple_files=True, type=["pdf"])

# Display the custom CSS
st.markdown(css, unsafe_allow_html=True)

# Document processing and vector database creation
if uploaded_files:
    if "final_documents" not in st.session_state:
        st.session_state["final_documents"] = []

    # Process uploaded files
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name

        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        document_chunks = text_splitter.split_documents(documents)

        for i, chunk in enumerate(document_chunks):
            chunk.metadata["source_document"] = uploaded_file.name

        st.session_state["final_documents"].extend(document_chunks)

    if st.session_state["final_documents"]:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        texts = [doc.page_content for doc in st.session_state["final_documents"]]
        metadatas = [doc.metadata for doc in st.session_state["final_documents"]]
        embeddings_list = embeddings.embed_documents(texts)
        st.session_state["vectors"] = FAISS.from_embeddings(
            text_embeddings=list(zip(texts, embeddings_list)),
            embedding=embeddings,
            metadatas=metadatas
        )
        st.sidebar.success("Vector database created successfully.")

# Define chatbot response logic
def chatbot_response(user_input, bot_id, previous_response=None):
    """Simulate a chatbot response with context and awareness of other bots."""
    # Extract only the message content for the chat history
    history = "\n".join(f"{speaker}: {message}" for speaker, message, _ in st.session_state["chat_history"])

    # Bot identity and references
    bot_name = f"AI_York_{bot_id}"
    other_bots = [f"AI_York_{i}" for i in range(1, 4) if i != bot_id]

    retriever = st.session_state["vectors"].as_retriever()
    retrieved_docs = retriever.get_relevant_documents(user_input)
    context = "\n".join(doc.page_content for doc in retrieved_docs[:5])

    prompt = f"""
    You are {bot_name}, part of a trio of AI assistants (AI_York_1, AI_York_2, AI_York_3). 
    Refer to yourself as {bot_name} and acknowledge the other assistants by their names ({', '.join(other_bots)}). 
    Take into account the following:
    - The user input: "{user_input}"
    - The chat history: "{history}"
    - The response of the previous assistant: "{previous_response}" (if available)
    - The retrieved context: "{context}"

    Respond briefly and conversationally, building on prior responses while introducing your unique perspective.
    """
    llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama3-70b-8192")
    response = llm.invoke([("user", prompt)]).content.strip()
    return response



# Main chat interface
st.title("Chat with Reflective Chatbots")

# Display custom CSS
st.markdown(css, unsafe_allow_html=True)

# Input section
user_input = st.text_input("Enter your input below:", "")

# Latest chat display at the top
if user_input and st.session_state["vectors"]:
    # Append the user input to the chat history
    st.session_state["chat_history"].insert(0, ("User", user_input, user_template))

    # Generate responses from each chatbot one by one
    bot_templates = [bot_template_1, bot_template_2, bot_template_3]
    bot_names = ["AI_York_1", "AI_York_2", "AI_York_3"]
    previous_response = None
    latest_responses = []

    for i in range(3):
        bot_name = bot_names[i]
        response = chatbot_response(user_input, bot_id=i + 1, previous_response=previous_response)
        latest_responses.append((bot_name, response, bot_templates[i]))
        st.session_state["chat_history"].insert(0, (bot_name, response, bot_templates[i]))
        previous_response = response
        # time.sleep(1)  # Simulate typing delay

    # Display the latest chat at the top in columns
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("### User")
        st.markdown(user_template.replace("{{MSG}}", user_input), unsafe_allow_html=True)

    for i, col in enumerate([col2, col3, col4]):
        bot_name, response, template = latest_responses[i]
        with col:
            st.markdown(f"### {bot_name}")
            st.markdown(template.replace("{{MSG}}", response), unsafe_allow_html=True)

# Scrollable past chat history at the bottom
if st.session_state["chat_history"]:
    with st.expander("Past Chat History"):
        for speaker, message, template in reversed(st.session_state["chat_history"]):
            st.markdown(template.replace("{{MSG}}", message), unsafe_allow_html=True)

elif not st.session_state["vectors"]:
    st.warning("Please upload documents to build the vector database.")