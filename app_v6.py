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

# Title
st.title("Discussion with my AI Clones")

# Display custom CSS
st.markdown(css, unsafe_allow_html=True)

# Input section
user_input = st.text_input("Enter your input below:", "")


# Initialize session states
if "vectors" not in st.session_state:
    st.session_state["vectors"] = None
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Sidebar for system prompt and uploading documents
with st.sidebar:
    st.title("Discussion with my AI Clones")
    uploaded_files = st.file_uploader("Upload PDF files", accept_multiple_files=True, type=["pdf"])
    
    # Editable system prompt
    default_prompt = (
        "York is a highly analytical and goal-oriented individual with a strong background in finance, data "
        "science, and artificial intelligence. He is taking a Master’s degree in MITB from SMU and has substantial "
        "professional experience, including leadership roles such as CFO at Decathlon Singapore. He excels in leveraging "
        "data for decision-making and thrives in exploring cutting-edge AI techniques to solve complex problems."
    )
    system_prompt = st.text_area("Edit System Prompt:", value=default_prompt, height=200)

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


# Function for chatbot response
def chatbot_response(user_input, bot_id, previous_response=None):
    """Simulate a chatbot response with meaningful use of search results and keyword-based logic."""
    # Extract chat history
    history = "\n".join(f"{speaker}: {message}" for speaker, message, _ in st.session_state["chat_history"])

    # Bot identity and references
    bot_names = ["Rag_York", "Wiki_York", "Search_York"]
    bot_name = bot_names[bot_id - 1]
    other_bots = [name for name in bot_names if name != bot_name]

    # Retrieve context from documents
    retriever = st.session_state["vectors"].as_retriever()
    retrieved_docs = retriever.get_relevant_documents(user_input)
    context = "\n".join(doc.page_content for doc in retrieved_docs[:5])

    # Format sources from retrieved documents
    document_sources = [
        f"**Document**: {doc.metadata['source_document']}\n**Excerpt**: {doc.page_content.strip()[:300]}..."
        for doc in retrieved_docs[:5]
    ]

    if bot_id == 1:
        # Rag_York: Retrieval-Augmented Generation (RAG) chatbot
        role_description = f"""
        You are {bot_name}, part of a trio of AI assistants (Rag_York, Wiki_York, Search_York). 
        Refer to yourself as {bot_name} and acknowledge the other assistants by their names ({', '.join(other_bots)}). 
        Your role is to retrieve and summarize information from the document database to assist in the discussion.
        """
        task_context = f"Retrieved Context: {context}"
        sources = document_sources
    elif bot_id in [2, 3]:
        # Use LLM to infer a meaningful keyword or phrase for search
        keyword_prompt = f"""
        Infer the most meaningful single keyword or phrase based on the user's input to use for a search query.
        
        User Input: "{user_input}"
        
        Provide only one keyword or keyphrase in your response, without additional text or punctuation.
        """
        llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama3-8b-8192")
        extracted_keyword = llm.invoke([("user", keyword_prompt)]).content.strip()

        # Handle cases where the LLM fails to provide a valid keyword
        if not extracted_keyword:
            extracted_keyword = "default"  # Fallback keyword
        search_results = []

        if bot_id == 2:
            # Wiki_York: Wikipedia search
            wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
            try:
                result = wiki_tool.run(extracted_keyword)
                if result:
                    search_results.append(f"**Wikipedia Result for '{extracted_keyword}'**: {result}")
                else:
                    search_results.append("No relevant Wikipedia results found.")
            except Exception as e:
                search_results.append(f"Error fetching data for '{extracted_keyword}': {str(e)}")
        elif bot_id == 3:
            # Search_York: Web search
            tool = BraveSearch.from_api_key(api_key=BRAVE_API_KEY, search_kwargs={"count": 1})
            try:
                result = tool.run(extracted_keyword)
                if isinstance(result, dict):
                    links = result.get("links", [])
                    if links:
                        search_results.append(f"**Web Link for '{extracted_keyword}'**: {links[0]}")
                    else:
                        search_results.append("No relevant web links found.")
                elif isinstance(result, list) and result:
                    search_results.append(f"**Web Link for '{extracted_keyword}'**: {result[0]}")
                else:
                    search_results.append(f"**Web Result for '{extracted_keyword}'**: {str(result)}")
            except Exception as e:
                search_results.append(f"Error fetching data for '{extracted_keyword}': {str(e)}")

        # Add role description and task context
        role_description = f"""
        You are {bot_name}, part of a trio of AI assistants (Rag_York, Wiki_York, Search_York). 
        Refer to yourself as {bot_name} and acknowledge the other assistants by their names ({', '.join(other_bots)}). 
        Your role is to perform a search using the inferred keyword: '{extracted_keyword}'.
        """
        task_context = f"""
        Retrieved Context: {context}
        Search Results: {', '.join(search_results)}
        """
        sources = document_sources + search_results

    else:
        return "Invalid bot ID."

    # Combine the role description, task context, and system prompt into the final prompt
    prompt = f"""
    {system_prompt}
    {role_description}
    Take into account the following:
    - The user input: "{user_input}"
    - The chat history: "{history}"
    - The response of the previous assistant: "{previous_response}" (if available)
    {task_context}
    Respond briefly and conversationally, building on prior responses while introducing your unique perspective.
    """
    llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama3-8b-8192")
    final_response = llm.invoke([("user", prompt)]).content.strip()

    return final_response, sources




# Main Chat Loop
if user_input and st.session_state.get("vectors"):
    st.session_state["chat_history"].insert(0, ("User", user_input, user_template))

    bot_templates = [bot_template_1, bot_template_2, bot_template_3]
    bot_names = ["Rag_York", "Wiki_York", "Search_York"]
    previous_response = None
    latest_responses = []

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("### York")
        st.markdown(user_template.replace("{{MSG}}", user_input), unsafe_allow_html=True)

    # Incremental chatbot responses
    for i, col in enumerate([col2, col3, col4]):
        bot_name = bot_names[i]
        response, sources = chatbot_response(user_input, bot_id=i + 1, previous_response=previous_response)
        latest_responses.append((bot_name, response, sources, bot_templates[i]))
        st.session_state["chat_history"].insert(0, (bot_name, response, bot_templates[i]))

        # Display the response and sources as they're generated
        with col:
            st.markdown(f"### {bot_name}")
            st.markdown(bot_templates[i].replace("{{MSG}}", response), unsafe_allow_html=True)
            with st.expander(f"{bot_name} - Sources"):
                for source in sources:
                    st.markdown(source)

        previous_response = response
        time.sleep(1)  # Simulate typing delay



# Scrollable past chat history at the bottom
if st.session_state.get("chat_history"):
    with st.expander("Past Chat History"):
        for speaker, message, template in reversed(st.session_state["chat_history"]):
            st.markdown(template.replace("{{MSG}}", message), unsafe_allow_html=True)

elif not st.session_state.get("vectors"):
    st.warning("Please upload documents to build the vector database.")


