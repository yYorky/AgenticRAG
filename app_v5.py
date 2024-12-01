# To improve in v6
# Display their sources (Rag top_k, wiki wiki page, search web links)
# Interesting header topic and attractive cover photo
# Improve prompting to behave and speak more like me as well and less like a helpful assistant, show some personality.

import os
import numpy as np
import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import BraveSearch

from dotenv import load_dotenv
import tempfile
from htmlTemplate import css, bot_template_1, bot_template_2, bot_template_3, user_template
import time

# Set up the Streamlit page
st.set_page_config(layout="wide")

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")


SYSTEM_PROMPT = """
You are one of three AI assistants named Rag_York, Wiki_York, and Search_York. Each assistant reflects the personality, expertise, and conversational style of a highly analytical and goal-oriented individual with a strong background in finance, data science, and artificial intelligence.

In this discussion:
1. Respond briefly, directly, and conversationally to keep the dialogue engaging.
2. Refer to the other assistants by their names (e.g., Wiki_York, Search_York) in your responses when appropriate.
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
    """Simulate a chatbot response with distinct roles for each AI bot."""
    # Extract only the message content for the chat history
    history = "\n".join(f"{speaker}: {message}" for speaker, message, _ in st.session_state["chat_history"])

    # Bot identity and references
    bot_names = ["Rag_York", "Wiki_York", "Search_York"]
    bot_name = bot_names[bot_id - 1]
    other_bots = [name for name in bot_names if name != bot_name]

    # Retrieve context from the document database for all bots
    retriever = st.session_state["vectors"].as_retriever()
    retrieved_docs = retriever.get_relevant_documents(user_input)
    context = "\n".join(doc.page_content for doc in retrieved_docs[:5])

    # Role-specific logic
    if bot_id == 1:
        # Rag_York: Retrieval-Augmented Generation (RAG) chatbot
        role_description = f"""
        You are {bot_name}, part of a trio of AI assistants (Rag_York, Wiki_York, Search_York). 
        Refer to yourself as {bot_name} and acknowledge the other assistants by their names ({', '.join(other_bots)}). 
        Your role is to retrieve and summarize information from the document database to assist in the discussion.
        """
        task_context = f"Retrieved Context: {context}"
    elif bot_id == 2:
        # Wiki_York: Wiki agent
        wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
        search_result = wiki_tool.run(user_input)
        role_description = f"""
        You are {bot_name}, part of a trio of AI assistants (Rag_York, Wiki_York, Search_York). 
        Refer to yourself as {bot_name} and acknowledge the other assistants by their names ({', '.join(other_bots)}). 
        Your role is to fetch and summarize information from Wikipedia, while also referencing the document context provided.
        """
        task_context = f"""
        Retrieved Context: {context}
        Wikipedia Search Result: {search_result}
        """
    elif bot_id == 3:
        # Search_York: Web search agent
        tool = BraveSearch.from_api_key(api_key=BRAVE_API_KEY, search_kwargs={"count": 3})
        search_result = tool.run(user_input)
        role_description = f"""
        You are {bot_name}, part of a trio of AI assistants (Rag_York, Wiki_York, Search_York). 
        Refer to yourself as {bot_name} and acknowledge the other assistants by their names ({', '.join(other_bots)}). 
        Your role is to search the web for relevant information while referencing the document context provided.
        """
        task_context = f"""
        Retrieved Context: {context}
        Web Search Result: {search_result}
        """
    else:
        return "Invalid bot ID."

    # Combine the role description and task context into the prompt
    prompt = f"""
    {role_description}
    Take into account the following:
    - The user input: "{user_input}"
    - The chat history: "{history}"
    - The response of the previous assistant: "{previous_response}" (if available)
    {task_context}

    Respond briefly and conversationally, building on prior responses while introducing your unique perspective.
    """

    # Generate response using LLM
    llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama3-70b-8192")
    final_response = llm.invoke([("user", prompt)]).content.strip()
    return final_response




# Main chat interface
st.title("Chat with Reflective Chatbots")

# Display custom CSS
st.markdown(css, unsafe_allow_html=True)

# Input section
user_input = st.text_input("Enter your input below:", "")

# Main chat loop
if user_input and st.session_state["vectors"]:
    st.session_state["chat_history"].insert(0, ("User", user_input, user_template))

    bot_templates = [bot_template_1, bot_template_2, bot_template_3]
    bot_names = ["Rag_York", "Wiki_York", "Search_York"]
    previous_response = None
    latest_responses = []

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("### User")
        st.markdown(user_template.replace("{{MSG}}", user_input), unsafe_allow_html=True)

    # Incremental chatbot responses
    for i, col in enumerate([col2, col3, col4]):
        bot_name = bot_names[i]
        response = chatbot_response(user_input, bot_id=i + 1, previous_response=previous_response)
        latest_responses.append((bot_name, response, bot_templates[i]))
        st.session_state["chat_history"].insert(0, (bot_name, response, bot_templates[i]))

        # Display the response as it's generated
        with col:
            st.markdown(f"### {bot_name}")
            st.markdown(bot_templates[i].replace("{{MSG}}", response), unsafe_allow_html=True)

        previous_response = response
        time.sleep(1)  # Simulate typing delay

# Scrollable past chat history at the bottom
if st.session_state["chat_history"]:
    with st.expander("Past Chat History"):
        for speaker, message, template in reversed(st.session_state["chat_history"]):
            st.markdown(template.replace("{{MSG}}", message), unsafe_allow_html=True)


elif not st.session_state["vectors"]:
    st.warning("Please upload documents to build the vector database.")
