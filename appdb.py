import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_objectbox.vectorstores import ObjectBox
from langchain_community.document_loaders import PyPDFDirectoryLoader
from dotenv import load_dotenv
import shutil

# Load environment variables
load_dotenv()

# Initialize API keys
os.environ['OPEN_API_KEY'] = os.getenv("OPENAI_API_KEY")
groq_api_key = os.getenv('GROQ_API_KEY')

# Streamlit page setup
st.set_page_config("TenderAI")
st.title("TenderAI: Smart solutions for tender queries")

# Initialize the Groq model
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Create the prompt template for Groq
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    <context>
    Questions:{input}
    """
)

# Initialize Vector Embedding and ObjectBox Database
def vector_embedding():
    st.session_state.embeddings = OpenAIEmbeddings()
    st.session_state.loader = PyPDFDirectoryLoader("./us_census")
    st.session_state.docs = st.session_state.loader.load()

    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)

    # **Clear old data before reloading**
    if "vectors" in st.session_state:
        del st.session_state["vectors"]  # Remove old vectors

    st.session_state.vectors = ObjectBox.from_documents(
        st.session_state.final_documents,
        st.session_state.embeddings,
        embedding_dimensions=768
    )

    st.success("Database updated! Try asking questions now.")

# Sidebar to show files and remove them
def file_management_sidebar():
    pdf_files = os.listdir('./us_census')

    if not pdf_files:
        st.sidebar.write("No files in the database.")
    else:
        st.sidebar.write("Current files in the database:")

        for pdf in pdf_files:
            # Button to remove the file
            col1, col2 = st.sidebar.columns([4, 1])
            col1.write(pdf)  # Display the file name
            if col2.button(f"❌", key=pdf):  # Cross icon button for removing
                file_path = os.path.join('./us_census', pdf)
                try:
                    os.remove(file_path)
                    st.sidebar.write(f"Successfully removed {pdf}.")
                    vector_embedding()  # Reinitialize the vector store after removing the file
                    st.experimental_rerun()  # Refresh the app to show the updated file list
                except Exception as e:
                    st.sidebar.write(f"Error removing {pdf}: {e}")

    # Allow the user to upload new files to the 'us_census' folder
    uploaded_files = st.sidebar.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)
    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Save uploaded PDF to the 'us_census' folder
            file_path = os.path.join('./us_census', uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.sidebar.write(f"Successfully added {uploaded_file.name}.")
        vector_embedding()  # Reinitialize the vector store after adding the new file
        st.experimental_rerun()  # Refresh the app to show the updated file list

# Call file management in sidebar
file_management_sidebar()

# Input for the user's query
input_prompt = st.text_input("Enter Your Question From Documents")

# Initialize the database and vectors
if st.button("Initialise"):
    vector_embedding()
    st.write("ObjectBox Database is ready")

import time

# Process user input if present
if input_prompt:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    start = time.process_time()

    response = retrieval_chain.invoke({'input': input_prompt})

    # Output the response and timing
    st.write(response['answer'])
    st.write("Response time:", time.process_time() - start)

    # Display similar documents using Streamlit expander
    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")
