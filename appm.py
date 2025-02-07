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
import time

load_dotenv()

# Load API keys
os.environ['OPEN_API_KEY'] = os.getenv("OPENAI_API_KEY")
groq_api_key = os.getenv('GROQ_API_KEY')

st.set_page_config("TenderAI")
st.title("TenderAI: Smart solutions for tender queries")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

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

# Vector Embedding and ObjectBox Vectorstore db
def vector_embedding():
    # Ensure old ObjectBox instance is closed
    if "vectors" in st.session_state and st.session_state["vectors"] is not None:
        st.session_state["vectors"].store.close()
        del st.session_state["vectors"]

    # Load documents and embeddings
    st.session_state.embeddings = OpenAIEmbeddings()
    st.session_state.loader = PyPDFDirectoryLoader("./us_census")
    st.session_state.docs = st.session_state.loader.load()

    # Text splitting
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)

    # Create ObjectBox instance (use a session-specific path)
    objectbox_path = f"./objectbox_data/session_{st.session_state.session_id}"
    os.makedirs(objectbox_path, exist_ok=True)

    st.session_state.vectors = ObjectBox.from_documents(
        st.session_state.final_documents,
        st.session_state.embeddings,
        embedding_dimensions=768,
        persist_directory=objectbox_path
    )

    st.success("Database updated! Try asking questions now.")

# Initialize session state for session tracking
if "session_id" not in st.session_state:
    st.session_state.session_id = str(int(time.time()))  # Unique session ID

input_prompt = st.text_input("Enter Your Question From Documents")

if st.button("Initialise"):
    vector_embedding()
    st.write("ObjectBox Database is ready")

if input_prompt and "vectors" in st.session_state:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    start = time.process_time()
    response = retrieval_chain.invoke({'input': input_prompt})
    st.write("Response time:", time.process_time() - start)
    st.write(response.get('answer', 'No answer found.'))

    # Display similar documents
    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response.get("context", [])):
            st.write(doc.page_content)
            st.write("--------------------------------")
