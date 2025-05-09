import os
import streamlit as st
from gpt4all import GPT4All
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer

# Initialize with progress bars
@st.cache_resource
def init_models():
    with st.spinner('Loading AI models...'):
        # Create models directory if needed
        os.makedirs("models", exist_ok=True)
        
        # Initialize models
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        llm = GPT4All(
            model_name="ggml-gpt4all-j-v1.3-groovy.bin",
            model_path="models/",
            allow_download=True
        )
        return embedding_model, llm

# App UI
st.title("📚 RAG Document Assistant")
embedding_model, llm = init_models()

# File Upload
uploaded_files = st.file_uploader("Upload documents", 
                                 type=['pdf', 'txt'], 
                                 accept_multiple_files=True)

# Process Documents
vector_store = None
if uploaded_files:
    documents = []
    for file in uploaded_files:
        with st.spinner(f'Processing {file.name}...'):
            file_path = f"temp_{file.name}"
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
            
            loader = PyPDFLoader(file_path) if file.name.endswith('.pdf') else TextLoader(file_path)
            documents.extend(loader.load())
            os.remove(file_path)  # Clean up
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    
    texts = [chunk.page_content for chunk in chunks]
    embeddings = embedding_model.encode(texts)
    vector_store = FAISS.from_embeddings(list(zip(texts, embeddings)), embedding_model)
    st.success(f"✅ Processed {len(chunks)} chunks from {len(uploaded_files)} files")

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            context = ""
            if vector_store:
                docs = vector_store.similarity_search(prompt, k=3)
                context = "\n\n".join([d.page_content for d in docs])
            
            response = llm.generate(f"Context: {context}\n\nQuestion: {prompt}\nAnswer:")
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})