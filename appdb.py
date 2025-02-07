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
import time
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


def cleanup_objectbox():
    """Ensure ObjectBox is properly closed before deletion"""
    if "vectors" in st.session_state:
        try:
            if hasattr(st.session_state.vectors, 'close'):
                st.session_state.vectors.close()  # Close database connection
            del st.session_state.vectors  # Remove from session state
        except Exception as e:
            st.warning(f"Error closing ObjectBox: {e}")

    # Wait to ensure all file handles are released
    time.sleep(2)  # Increase delay to allow Windows to release the file lock

    # Attempt deletion safely
    if os.path.exists("objectbox"):
        try:
            shutil.rmtree("objectbox")
            st.info("ObjectBox database successfully cleaned up.")
        except PermissionError as e:
            st.error(f"")
#Failed to delete ObjectBox directory: {e}. Try restarting the app.
# Initialize Vector Embedding and ObjectBox Database
def vector_embedding():
    cleanup_objectbox()  # Clean up before creating new store

    st.session_state.embeddings = OpenAIEmbeddings()
    st.session_state.loader = PyPDFDirectoryLoader("./us_census")
    st.session_state.docs = st.session_state.loader.load()

    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)

    if os.path.exists("objectbox"):
        st.session_state.vectors = ObjectBox(st.session_state.embeddings, "objectbox")
    else:
        st.session_state.vectors = ObjectBox.from_documents(
            st.session_state.final_documents,
            st.session_state.embeddings,
            embedding_dimensions=768
        )


    st.success("Database updated! Try asking questions now.")

def file_management_sidebar():
    if "pdf_files" not in st.session_state:
        st.session_state.pdf_files = os.listdir('./us_census')

    if not st.session_state.pdf_files:
        st.sidebar.write("No files in the database.")
    else:
        st.sidebar.write("Current files in the database:")

        files_to_keep = []  # Temporary list to store remaining files

        for idx, pdf in enumerate(st.session_state.pdf_files):  # Use index for unique keys
            col1, col2 = st.sidebar.columns([4, 1])
            col1.write(pdf)  # Display the file name
            if col2.button(f"x", key=f"remove_{idx}_{pdf}"):  # Unique key using index and filename
                file_path = os.path.join('./us_census', pdf)
                try:
                    os.remove(file_path)
                    st.sidebar.write(f"Successfully removed {pdf}.")
                    
                    # Remove the file from session_state list
                    st.session_state.pdf_files.remove(pdf)
                    
                    vector_embedding()  # Reinitialize vector store after deletion
                    st.rerun()  # Force UI refresh
                except Exception as e:
                    st.sidebar.write(f"Error removing {pdf}: {e}")
                    files_to_keep.append(pdf)
                    st.rerun()
            else:
                files_to_keep.append(pdf)

        # Update session state with remaining files
        st.session_state.pdf_files = files_to_keep

    # Allow file uploads
    uploaded_files = st.sidebar.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)
    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_path = os.path.join('./us_census', uploaded_file.name)
            
            # Check if file already exists before writing
            if uploaded_file.name not in st.session_state.pdf_files:
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                st.sidebar.write(f"Successfully added {uploaded_file.name}.")
                st.session_state.pdf_files.append(uploaded_file.name)  # Update session state
            else:
                st.sidebar.write(f"{uploaded_file.name} is already in the database.")
            break

        vector_embedding()  # Reinitialize vector store
        st.rerun()  # Refresh UI


# Call file management in sidebar
file_management_sidebar()

# Input for the user's query
input_prompt = st.text_input("Enter Your Question From Documents")

# Initialize the database and vectors
if st.button("Initialise"):
    #cleanup_objectbox()  # Ensure clean state before initialization
    vector_embedding()
    st.write("ObjectBox Database is ready")

import time

# Process user input if present
if input_prompt:
    try:
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
    except AttributeError:
        st.error("Database not initialized. Please click the Initialise button first.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")