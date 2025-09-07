import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import faiss
import PyPDF2
import numpy as np
import torch
import os
import time

# ------------------ Load Pretrained Models ------------------
# Load QA model from Hugging Face for question answering
qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
# Load Summarizer
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
# Load Sentiment model
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
# Load NER model
ner_model = pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True)

# Load Sentence-Transformers model for embedding chunks
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# ------------------ File Upload ------------------
def getText():
    uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=['pdf', 'txt'])
    if uploaded_file is None:
        return None

    if uploaded_file.name.lower().endswith('pdf'):
        reader = PyPDF2.PdfReader(uploaded_file)
        content = ''
        for page in reader.pages:
            content += page.extract_text()
    elif uploaded_file.name.lower().endswith('txt'):
        content = uploaded_file.read().decode('utf-8')
    else:
        st.error("Please upload a PDF or TXT file.")
        content = None
    return content

# ------------------ Summarization ------------------
def summarize(text):
    summaryResult = summarizer(text, max_length=150, min_length=30, do_sample=False)
    return summaryResult[0]['summary_text']

# ------------------ Sentiment Analysis ------------------
def analyzeSentiment(text):
    sentimentResult = sentiment_analyzer(text[:512])  # Limiting to the first 512 tokens
    return sentimentResult[0]

# ------------------ Named Entity Recognition ------------------
def findEntities(text):
    entitiesResult = ner_model(text[:1000])  # Limiting to the first 1000 tokens
    return entitiesResult

# ------------------ Chunking the Document ------------------
def chunk_text(text, chunk_size=500, overlap=50):
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    tokens = tokenizer.tokenize(text)

    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk = tokenizer.convert_tokens_to_string(tokens[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

# ------------------ FAISS and Embedding ------------------
def embed_chunks(chunks):
    embeddings = embedding_model.encode(chunks)
    return embeddings

def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)  # Build FAISS index for the embeddings
    index.add(embeddings)
    return index

def retrieve_chunks(question, chunks, index, top_k=3):
    question_embedding = embedding_model.encode([question])
    distances, indices = index.search(np.array(question_embedding), top_k)
    retrieved_chunks = [chunks[i] for i in indices[0]]
    return retrieved_chunks

# ------------------ RAG Answer Generation ------------------
def rag_answer(question, retrieved_chunks):
    context = " ".join(retrieved_chunks)
    input_text = f"question: {question} context: {context}"
    output = qa_model(input_text, max_length=150, min_length=30, do_sample=False)
    return output['answer']

# ------------------ Main Function ------------------
def main():
    st.title("Document Analysis Tool with Question Answering (RAG)")

    uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=['pdf', 'txt'])
    if uploaded_file is not None:
        content = getText()

        if content:
            # ------------------ Document Content Display ------------------
            st.subheader("Document Content:")
            st.write(content[:1000] + "...")  # Display the first 1000 characters of the document

            # ------------------ Summarization ------------------
            summary = summarize(content)
            st.subheader("Document Summary:")
            st.write(summary)

            # ------------------ Sentiment Analysis ------------------
            sentiment = analyzeSentiment(content)
            st.subheader("Sentiment Analysis:")
            st.write(f"Sentiment Label: {sentiment['label']}, with a confidence score of {sentiment['score']:.4f}")

            # ------------------ Named Entity Recognition ------------------
            entities = findEntities(content)
            st.subheader("Named Entities:")
            for entity in entities:
                st.write(f"Entity: {entity['word']}, Label: {entity['entity_group']}")

            # ------------------ Question Answering ------------------
            question = st.text_input("Ask a question about the document:")

            if question:
                # Step 1: Chunk the text into smaller sections
                chunks = chunk_text(content)

                # Step 2: Create embeddings for the chunks
                embeddings = embed_chunks(chunks)

                # Step 3: Build the FAISS index for efficient retrieval
                index = build_faiss_index(np.array(embeddings))

                # Step 4: Retrieve the most relevant chunks for the question
                retrieved_chunks = retrieve_chunks(question, chunks, index)

                if len(retrieved_chunks) == 0:
                    st.error("No relevant chunks found. Try asking a different question.")
                else:
                    # Step 5: Generate an answer using the retrieved chunks
                    answer = rag_answer(question, retrieved_chunks)
                    st.write("**Answer:**", answer)

if __name__ == "__main__":
    main()
