# Chat with PDF - Q&A Chatbot

## Overview

The "Chat with PDF" system is a Streamlit-based application that allows users to upload PDF documents and interact with a Question-and-Answer chatbot powered by Groq API and state-of-the-art language models. The chatbot retrieves relevant information from uploaded PDFs and provides detailed responses to user queries.

---

## Features

- **PDF Upload**: Upload multiple PDF documents for content analysis.
- **Vector Database**: Automatically process and store document embeddings for efficient retrieval.
- **Q&A Functionality**: Interact with an intelligent chatbot that answers questions based on the content of the uploaded PDFs.
- **Performance Tracking**: Includes response time calculation for every query.

---

## Tech Stack

### 1. **Frontend**
   - [Streamlit](https://streamlit.io/): Simplifies the UI creation process for Python-based applications.

### 2. **Backend and API Integration**
   - **Groq API**:
     - **Model**: `Llama3-8b-8192` for natural language processing.
     - **Purpose**: Provides powerful conversational capabilities.
   - **LangChain Ecosystem**:
     - `langchain_groq`: For seamless integration with the Groq API.
     - `langchain_community.vectorstores.Chroma`: To manage vector embeddings for document retrieval.
     - `langchain_huggingface.HuggingFaceEmbeddings`: Embeddings generated using the `all-MiniLM-L6-v2` model.

### 3. **Storage and Processing**
   - **Chroma**: Used for embedding storage with support for document similarity retrieval.
   - **DuckDB**: Backend database for the Chroma vector store.

### 4. **Document Handling**
   - **PyPDFDirectoryLoader**: Loads and parses PDF files.
   - **CharacterTextSplitter**: Splits documents into smaller chunks for efficient embedding and retrieval.

### 5. **Prompt Engineering**
   - **Custom Prompt Template**: Designed for a context-aware chatbot leveraging Groq's conversational abilities.

---

## Workflow

1. **PDF Upload**:
   - Users upload one or more PDF files via the Streamlit sidebar.
   - Files are stored locally for processing.

2. **Document Processing**:
   - PDFs are parsed using `PyPDFDirectoryLoader`.
   - Text is split into chunks using `CharacterTextSplitter` to prepare for embedding.

3. **Vector Embedding**:
   - Document embeddings are generated using `HuggingFaceEmbeddings`.
   - The Chroma vector database stores these embeddings for efficient retrieval.

4. **Chat Interaction**:
   - Users type queries into the chatbot input field.
   - The chatbot retrieves relevant document chunks using Chroma's retriever.
   - Groq API processes the prompt and returns the answer.

---

## Installation and Setup

### Prerequisites
- Python 3.8 or higher
- Required libraries:
  ```bash
  pip install streamlit langchain chromadb duckdb pyarrow langchain-huggingface
  ```
- Groq API Key (to be provided in the Streamlit sidebar).

### Steps
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/chat-with-pdf.git
   cd chat-with-pdf
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
4. Upload your PDF files and interact with the chatbot!

---

## Usage Instructions

1. Open the application.
2. Enter your Groq API key in the sidebar.
3. Upload one or more PDF files.
4. Click "Document Embedding" to process the PDFs.
5. Ask questions in the chat input field.
6. View answers and related documents.

---

## Example Interaction

**User**: "What is the main topic of the first uploaded document?"

**Chatbot**: "The main topic of the document is [extracted content based on the document]."

---

## Future Enhancements
- Add support for more file types (e.g., Word, Excel).
- Implement advanced search and filtering options.
- Improve response streaming for real-time interaction.

---

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

---

## Acknowledgments

- [Streamlit](https://streamlit.io/)
- [LangChain](https://langchain.com/)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Chroma](https://docs.trychroma.com/)
- [Groq API](https://groq.com/)

---

Feel free to contribute or raise issues for further development!

