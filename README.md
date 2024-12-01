# AGENTIC RAG - Let's discuss with my AI clones 🤖🗂️

![Chatbot Banner](https://raw.githubusercontent.com/yYorky/AgenticRAG/refs/heads/main/static/Clones%20York.png)

## Motivation ✨

**Discussion with My AI Clones** is a Streamlit-based web application that enables dynamic, context-aware conversations using three specialized AI assistants. The assistants—Rag_York, Wiki_York, and Search_York—leverage advanced Retrieval-Augmented Generation (RAG), Wikipedia search, and web search capabilities to provide meaningful insights tailored to user queries.

This application is ideal for professionals, researchers, and enthusiasts who want to explore cutting-edge conversational AI while interacting with their uploaded documents or external information sources.

---

## Features 🧰

![Chatbot Banner](https://raw.githubusercontent.com/yYorky/AgenticRAG/refs/heads/main/static/AI%20Clones%20Chat.JPG)

- **Three Specialized AI Clones**:
  - **Rag_York**: Uses RAG to retrieve and summarize content from uploaded PDF documents.
  - **Wiki_York**: Searches Wikipedia for relevant information using inferred keywords from user input.
  - **Search_York**: Performs web searches to enrich responses with real-time information.

- **Editable System Prompt**: Tailor the clones' behavior with a prefilled prompt editable via the sidebar.

- **PDF Document Support**: Upload multiple PDF files for processing and vectorized retrieval.

- **Responsive Chat Interface**:
  - Incremental chatbot responses.
  - Expandable "Sources" section for each response, showing the context and references used.

- **Keyword Inference**: AI-powered extraction of meaningful keywords for precise Wikipedia and web searches.

---

## User Interface 🖥️

### Sidebar
- **PDF Upload**: Load one or more PDF documents for context-aware retrieval.
- **System Prompt**: Edit the behavior of AI clones using a prefilled system prompt.

### Main Chat
- **Dynamic Responses**: View responses from each AI clone, displayed incrementally.
- **Expandable Sources**: Check the context used for generating replies, including document excerpts, Wikipedia summaries, and web links.
- **Chat History**: Review previous conversations in a scrollable section.

---

## Project Structure 📂

```plaintext
AGENTIC RAG/
├── __pycache__/                  # Compiled Python files
├── agent clone archive/          # Archived versions of agent clone implementations
│   ├── app_v4.py
│   ├── app_v5.py
├── agentic rag app archive/      # Archived versions of main app implementations
│   ├── app_v1.py
│   ├── app_v2.py
│   ├── app_v3.py
├── static/                       # Static assets for UI
│   ├── York_AI_1.png
│   ├── York_AI_2.png
│   ├── York_AI_3.png
│   ├── York.jpg
├── .gitignore                    # Git ignore file for sensitive or unnecessary files
├── app_v6.py                     # Main Streamlit app file
├── htmlTemplate.py               # CSS and HTML templates for chat layout
├── requirements.txt              # Python dependencies

```

---

## Setup Instructions 📋

### Prerequisites ✅

- Python 3.10 or higher
- Streamlit installed (`pip install streamlit`)
- API keys for Groq (https://console.groq.com/login) and Brave Search (https://api.search.brave.com/)

---

### Step-by-Step Instructions 🔢

1. 📥 **Clone the repository**
   ```bash
   git clone https://github.com/your_username/discussion_ai_clones.git
   cd discussion_ai_clones
   ```

2. 🐍 **Set up a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. 📦 **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. 🛠️ **Set up environment variables**
   - Create a `.env` file in the root directory.
   - Add your API keys:
     ```env
     GROQ_API_KEY=your_groq_api_key
     BRAVE_API_KEY=your_brave_api_key
     ```

5. 🏃 **Run the Streamlit app**
   ```bash
   streamlit run app_v6.py
   ```

---

## Usage 🕹️

1. Open the app in your browser.
2. Upload one or more PDF documents via the sidebar.
3. Enter your query in the input box.
4. Review responses incrementally from:
   - **Rag_York**: Contextual retrieval from uploaded PDFs.
   - **Wiki_York**: Wikipedia-based insights using inferred keywords.
   - **Search_York**: Web search results using keyword inference.
5. Expand the "Sources" section under each response for detailed references.

---

## How It Works 🔍

1. **PDF Processing**:
   - Uploaded PDFs are split into manageable chunks.
   - FAISS vector embeddings enable efficient similarity searches.

2. **Keyword Inference**:
   - For Wiki_York and Search_York, an LLM generates meaningful keywords based on user input.

3. **Dynamic Retrieval**:
   - Rag_York fetches document excerpts.
   - Wiki_York queries Wikipedia for the inferred keyword.
   - Search_York performs web searches with Brave Search.

4. **Chat Interface**:
   - Responses are displayed incrementally.
   - Sources and search results are shown in an expandable section.

---

## Contributing 🤝

We welcome contributions! Fork the repository, create a feature branch, and submit a pull request.
