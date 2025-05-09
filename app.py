# Add this at the very top of app.py
try:
    import huggingface_hub
    if huggingface_hub.__version__ != '0.16.4':
        raise ImportError("Wrong huggingface-hub version")
except ImportError:
    raise ImportError("Please install huggingface-hub==0.16.4")

# Core imports
import os
import re
import logging
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS

# Other ML imports
from sentence_transformers import SentenceTransformer
from gpt4all import GPT4All
import math
import time
from typing import List, Dict, Optional

# PDF loader fallback
try:
    from langchain_community.document_loaders import PyPDFLoader
except ImportError:
    raise ImportError("Please install pypdf: pip install pypdf")

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'txt'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize models
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
model_path = "C:\\Users\\SURABHI\\AppData\\Local\\nomic.ai\\GPT4All\\mistral-7b-openorca.gguf2.Q4_0.gguf"
llm = GPT4All(model_path, allow_download=False)


# Initialize vector store
vector_store = None

# Configure logging
logging.basicConfig(
    filename='agent.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class DocumentProcessor:
    @staticmethod
    def allowed_file(filename: str) -> bool:
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

    @staticmethod
    def load_and_process_documents(file_paths: List[str]) -> List[str]:
        documents = []
        for file_path in file_paths:
            try:
                if file_path.endswith('.pdf'):
                    loader = PyPDFLoader(file_path)
                else:
                    loader = TextLoader(file_path)
                documents.extend(loader.load())
            except Exception as e:
                logging.error(f"Error loading {file_path}: {str(e)}")
                continue
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        return chunks

    @staticmethod
    def create_vector_store(chunks: List[str]) -> FAISS:
        texts = [chunk.page_content for chunk in chunks]
        embeddings = embedding_model.encode(texts, show_progress_bar=True)
        
        global vector_store
        vector_store = FAISS.from_embeddings(
            text_embeddings=list(zip(texts, embeddings)),
            embedding=embedding_model
        )
        return vector_store

class QueryProcessor:
    @staticmethod
    def retrieve_relevant_chunks(query: str, k: int = 3) -> List[str]:
        if vector_store is None:
            return []
        
        query_embedding = embedding_model.encode([query])
        docs = vector_store.similarity_search_by_vector(query_embedding[0], k=k)
        return [doc.page_content for doc in docs]

    @staticmethod
    def generate_answer(query: str, context: str) -> str:
        prompt = f"""You are a helpful AI assistant. Answer the question based on the context provided. 
        If the answer isn't in the context, say you don't know. Be concise but informative.

Context:
{context}

Question: {query}
Answer:"""
        
        try:
            response = llm.generate(prompt, max_tokens=1500, temp=0.7)
            return response.strip()
        except Exception as e:
            logging.error(f"LLM generation error: {str(e)}")
            return "I encountered an error while generating an answer."

    @staticmethod
    def calculate_expression(expression: str) -> str:
        try:
            safe_expr = re.sub(r'[^0-9+\-*/(). ]', '', expression)
            result = eval(safe_expr, {'__builtins__': None}, {'math': math})
            return str(result)
        except Exception as e:
            logging.error(f"Calculation error: {str(e)}")
            return "I couldn't calculate that expression."

    @staticmethod
    def define_term(term: str) -> str:
        definitions = {
            "algorithm": "A set of rules or steps used to solve a problem or perform a computation.",
            "api": "Application Programming Interface - a set of protocols for building and integrating software.",
            "database": "An organized collection of structured information or data.",
            "rag": "Retrieval-Augmented Generation - combines information retrieval with text generation.",
            "llm": "Large Language Model - an AI model trained on vast amounts of text data."
        }
        return definitions.get(term.lower(), f"I don't have a definition for '{term}'.")

    @staticmethod
    def route_query(query: str) -> Dict:
        logging.info(f"Routing query: {query}")
        
        # Check for calculation requests
        if any(word in query.lower() for word in ['calculate', 'compute', 'math', 'solve', 'what is']):
            match = re.search(r'([\d+\-*/(). ]+)', query)
            if match:
                expression = match.group(1)
                result = QueryProcessor.calculate_expression(expression)
                logging.info(f"Used calculator for expression: {expression}")
                return {
                    "tool": "calculator",
                    "result": result,
                    "context": None
                }
        
        # Check for definition requests
        if any(word in query.lower() for word in ['define', 'definition', 'what is a']):
            match = re.search(r'(?:define|what is a?) (.+?)(?:\?|$)', query.lower())
            if match:
                term = match.group(1)
                result = QueryProcessor.define_term(term)
                logging.info(f"Used dictionary for term: {term}")
                return {
                    "tool": "dictionary",
                    "result": result,
                    "context": None
                }
        
        # Default to RAG pipeline
        context_chunks = QueryProcessor.retrieve_relevant_chunks(query)
        context = "\n\n".join(context_chunks) if context_chunks else "No relevant context found."
        answer = QueryProcessor.generate_answer(query, context)
        
        logging.info(f"Used RAG pipeline with {len(context_chunks)} context chunks")
        return {
            "tool": "RAG",
            "result": answer,
            "context": context_chunks
        }

@app.route('/')
def home():
    return render_template('chat.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'files[]' not in request.files:
        return jsonify({'error': 'No files uploaded'}), 400
    
    files = request.files.getlist('files[]')
    file_paths = []
    
    for file in files:
        if file and DocumentProcessor.allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            try:
                file.save(file_path)
                file_paths.append(file_path)
            except Exception as e:
                logging.error(f"Error saving file {filename}: {str(e)}")
                continue
    
    if not file_paths:
        return jsonify({'error': 'No valid files uploaded'}), 400
    
    try:
        chunks = DocumentProcessor.load_and_process_documents(file_paths)
        DocumentProcessor.create_vector_store(chunks)
        return jsonify({
            'message': f'Successfully processed {len(chunks)} document chunks',
            'num_chunks': len(chunks)
        })
    except Exception as e:
        logging.error(f"Document processing error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    query = data.get('query', '').strip()
    
    if not query:
        return jsonify({'error': 'Empty query'}), 400
    
    start_time = time.time()
    response = QueryProcessor.route_query(query)
    response_time = time.time() - start_time
    
    logging.info(f"Query processed in {response_time:.2f} seconds")
    response['response_time'] = response_time
    return jsonify(response)

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # <-- Add this line
    app.run(host='0.0.0.0', port=port, debug=False)  # <-- Modified line