import streamlit as st
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import faiss
import PyPDF2
import numpy as np

# ------------------ Load Pre-trained Models ------------------
qa_generator = pipeline("text2text-generation", model="facebook/bart-large-cnn")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# ------------------ Helper Functions ------------------

# Function to chunk the text into smaller sections
def chunk_text(text, chunk_size=500):
    tokens = text.split()
    chunks = [" ".join(tokens[i:i + chunk_size]) for i in range(0, len(tokens), chunk_size)]
    return chunks

# Function to embed chunks of text
def embed_chunks(chunks):
    embeddings = embedding_model.encode(chunks)
    return embeddings

# Function to build the FAISS index
def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)  # Build FAISS index
    index.add(embeddings)
    return index

# Function to retrieve relevant chunks based on the question
def retrieve_chunks(question, chunks, index, top_k=3):
    question_embedding = embedding_model.encode([question])
    distances, indices = index.search(np.array(question_embedding), top_k)
    return [chunks[i] for i in indices[0]]

# Function to generate an answer using BART
def rag_answer(question, retrieved_chunks):
    context = " ".join(retrieved_chunks)
    input_text = f"question: {question} context: {context}"
    output = qa_generator(input_text, max_length=150, min_length=30, do_sample=False)
    return output[0]['generated_text']

# ------------------ Streamlit UI ------------------

# Streamlit file uploader
st.title("Document Question Answering with RAG")

uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=['pdf', 'txt'])

if uploaded_file is not None:
    # Extract text from the uploaded file
    if uploaded_file.name.lower().endswith('pdf'):
        reader = PyPDF2.PdfReader(uploaded_file)
        content = ''
        for page in reader.pages:
            content += page.extract_text()
    elif uploaded_file.name.lower().endswith('txt'):
        content = uploaded_file.read().decode('utf-8')

    st.subheader("Document Content:")
    st.write(content[:1000] + "...")  # Display the first 1000 characters of the content

    # Step 1: Chunk the document into smaller sections
    chunks = chunk_text(content)

    # Step 2: Generate embeddings for the chunks
    embeddings = embed_chunks(chunks)

    # Step 3: Build the FAISS index
    index = build_faiss_index(np.array(embeddings))
    st.write("FAISS index created successfully.")

    # Step 4: Ask the user for a question
    question = st.text_input("Ask a question about the document:")

    if question:
        # Step 5: Retrieve relevant chunks from the FAISS index
        retrieved_chunks = retrieve_chunks(question, chunks, index)

        if len(retrieved_chunks) == 0:
            st.error("No relevant chunks found. Try asking a different question.")
        else:
            # Step 6: Generate an answer using the retrieved chunks
            answer = rag_answer(question, retrieved_chunks)
            st.write("**Answer:**", answer)
