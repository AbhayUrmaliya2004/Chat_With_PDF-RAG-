import streamlit as st
import os
import time
from langchain_groq import ChatGroq
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings

# Sidebar and title
st.sidebar.title("Settings")
st.title("Q&A Chatbot")
groq_api_key = st.sidebar.text_input("Groq API Key", type="password")

def format_docs(docs):
    """Format the top documents for display."""
    return "\n\n".join(doc.page_content for doc in docs)

def create_vector_embedding():
    """Create embeddings and initialize the vector store."""
    if "vectorstore" not in st.session_state:
        # Load embeddings
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        st.session_state.loader = PyPDFDirectoryLoader("4-Document_QnA_with_GROQ/Data")
        st.session_state.docs = st.session_state.loader.load()

        # Split documents
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        st.session_state.final_documents = text_splitter.split_documents(st.session_state.docs)
       
        # Create vector database
        st.session_state.vectorstore = FAISS.from_documents(
            st.session_state.final_documents,
            embedding=st.session_state.embeddings
        )
        # st.write("Vector Database Created Successfully!")  # Optional debug log

def initialize_qa_components():
    """Initialize and cache LLM, prompt, and QA chain."""
    if "qa_chain" not in st.session_state:
        # Initialize LLM
        st.session_state.llm = ChatGroq(model_name="Llama3-8b-8192", groq_api_key=groq_api_key)
        
        # Initialize prompt template
        st.session_state.rag_prompt = ChatPromptTemplate.from_template(
            """
            You are a conversational Question Answer chatbot who answers based on the 
            provided documents as PDFs. 
            <context>
            {context}
            <context>
            Question: {input}
            """
        )

        # Create chain
        retriever = st.session_state.vectorstore.as_retriever()
        retriever.search_kwargs = {"k": 4}  # Retrieve top 4 documents
        document_chain = create_stuff_documents_chain(st.session_state.llm, st.session_state.rag_prompt)
        st.session_state.qa_chain = create_retrieval_chain(retriever, document_chain)
        # st.write("QA Components Initialized Successfully!")  # Optional debug log

if groq_api_key:
    # Upload PDFs
    pdf_files = st.sidebar.file_uploader("Upload your PDF", type="pdf", accept_multiple_files=True)

    if st.sidebar.button("Upload PDFs"):
        if pdf_files:
            os.makedirs("4-Document_QnA_with_GROQ/Data", exist_ok=True)
            for uploaded_file in pdf_files:
                with open(f"4-Document_QnA_with_GROQ/Data/{uploaded_file.name}", "wb") as f:
                    f.write(uploaded_file.getbuffer())
            st.sidebar.success("Files Uploaded Successfully!")
        else:
            st.sidebar.error("Please upload at least one PDF.")

    # Document Embedding
    if st.sidebar.button("Document Embedding"):
        try:
            create_vector_embedding()
            st.sidebar.success("Embeddings Created Successfully!")
        except Exception as e:
            st.sidebar.error(f"Error in embedding creation: {e}")

    # Chat Section
    if user_prompt := st.chat_input(placeholder="Ask anything from the uploaded PDF"):
        if "vectorstore" not in st.session_state:
            st.error("Please create document embeddings first.")
        else:
            try:
                # Initialize QA components if not already done
                initialize_qa_components()
                
                # Retrieve and format documents
                retriever = st.session_state.vectorstore.as_retriever()
                docs = retriever.get_relevant_documents(user_prompt)
                formatted_context = format_docs(docs)
                
                # Process the question
                start_time = time.process_time()
                response = st.session_state.qa_chain.invoke({"input": user_prompt, "context": formatted_context})
                elapsed_time = time.process_time() - start_time

                st.chat_message("user").write(user_prompt)
                # Display the response in the chat
                with st.chat_message("AI Assistant"):
                    st.markdown(response["answer"])

            except Exception as e:
                st.error(f"Error during question answering: {e}")
