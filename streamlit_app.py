import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import faiss
import PyPDF2
import torch
import os
import numpy as np
import transformers

# Silence warnings & tokenizer crash
transformers.logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# -------------------------
# Cached Models
# -------------------------
@st.cache_resource(show_spinner=False)
def get_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource(show_spinner=False)
def get_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

@st.cache_resource(show_spinner=False)
def get_sentiment_model():
    return pipeline("sentiment-analysis")

@st.cache_resource(show_spinner=False)
def get_ner_model():
    return pipeline("ner", grouped_entities=True)

@st.cache_resource(show_spinner=False)
def get_qa_model():
    return pipeline("question-answering")

@st.cache_resource(show_spinner=False)
def get_qa_generator():
    return pipeline("text2text-generation", model="sshleifer/distilbart-cnn-12-6")

@st.cache_resource(show_spinner=False)
def get_tokenizer_and_model():
    model_path = './fine_tuned_model'
    if os.path.exists(model_path):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        st.success("Loaded fine-tuned model.")
    else:
        tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
        model = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-12-6")
    return tokenizer, model

# -------------------------
# File Upload
# -------------------------
def getText():
    uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=['pdf', 'txt'])
    if uploaded_file is None:
        return None

    if uploaded_file.name.lower().endswith('pdf'):
        reader = PyPDF2.PdfReader(uploaded_file)
        content = ''
        for page in reader.pages:
            text = page.extract_text()
            if text:
                content += text
    elif uploaded_file.name.lower().endswith('txt'):
        content = uploaded_file.read().decode('utf-8')
    else:
        st.error("Please upload a PDF or TXT file.")
        content = None
    return content

# -------------------------
# Summarization
# -------------------------
def summarize(text):
    summarizer = get_summarizer()
    summaryResult = summarizer(text, max_length=150, min_length=30, do_sample=False)
    st.write("### Summary:")
    st.write(summaryResult[0]['summary_text'])

    userSummary = st.text_input("Is this summary good? If not, type your improved summary here. Otherwise, leave blank:")
    return summaryResult[0]['summary_text'], userSummary

# -------------------------
# Sentiment
# -------------------------
def analyzeSentiment(text):
    sentimentModel = get_sentiment_model()
    sentimentResult = sentimentModel(text[:512])
    st.write("### Sentiment of the text:")
    st.write(sentimentResult[0])
    return sentimentResult[0]

# -------------------------
# NER
# -------------------------
def findEntities(text):
    nerModel = get_ner_model()
    entitiesResult = nerModel(text[:1000])
    st.write("### Named entities found:")
    st.write(entitiesResult)
    return entitiesResult

# -------------------------
# Main Topic
# -------------------------
def findMainTopic(text):
    qaModel = get_qa_model()
    question = "What is the main topic of the document?"
    answer = qaModel(question=question, context=text[:1000])
    st.write("### Main topic:")
    st.write(answer['answer'])
    return answer['answer']

# -------------------------
# RAG Helpers
# -------------------------
def chunk_text(text, chunk_size=500, overlap=50):
    tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
    tokens = tokenizer.tokenize(text)

    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk = tokenizer.convert_tokens_to_string(tokens[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def embed_chunks(chunks):
    embedding_model = get_embedding_model()
    embeddings = embedding_model.encode(chunks)
    return np.array(embeddings).astype("float32")   # FIXED for FAISS

def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def retrieve_chunks(question, chunks, index, top_k=3):
    question_embedding = get_embedding_model().encode([question]).astype("float32")
    distances, indices = index.search(question_embedding, top_k)
    retrieved_chunks = [chunks[i] for i in indices[0]]
    return retrieved_chunks

def rag_answer(question, retrieved_chunks):
    qa_generator = get_qa_generator()
    context = " ".join(retrieved_chunks)
    input_text = f"question: {question} context: {context}"
    output = qa_generator(input_text, max_length=150, min_length=30, do_sample=False)
    return output[0]['generated_text']

def askQuestionsRAG(text):
    chunks = chunk_text(text)
    embeddings = embed_chunks(chunks)
    index = build_faiss_index(embeddings)

    st.write("\nYou can now ask questions based on the document. Type your question and press enter.")
    question = st.text_input("Ask a question or leave blank to stop:", key="rag_question")
    if question:
        try:
            retrieved_chunks = retrieve_chunks(question, chunks, index)
            answer = rag_answer(question, retrieved_chunks)
            st.write("**Answer:**", answer)
        except Exception as e:
            st.exception(e)   # FIXED: show full error

# -------------------------
# Fine-tuning (disabled on Cloud)
# -------------------------
def fineTune(text, improvedSummary):
    st.warning("⚠ Fine-tuning disabled on Streamlit Cloud due to resource limits. Run locally instead.")
    return

# -------------------------
# Main Run
# -------------------------
def run():
    state = {}
    state['content'] = getText()
    if state['content']:
        state['summary'], state['userSummary'] = summarize(state['content'])
        state['sentiment'] = analyzeSentiment(state['content'])
        state['entities'] = findEntities(state['content'])
        state['topic'] = findMainTopic(state['content'])
        askQuestionsRAG(state['content'])
        fineTune(state['content'], state['userSummary'])

if __name__ == "__main__":
    run()
