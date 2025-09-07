import streamlit as st
import PyPDF2
import os
import torch
import numpy as np
import faiss

from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from sentence_transformers import SentenceTransformer

# ---------------------------
# File Text Extraction
# ---------------------------
def get_text(uploaded_file):
    if uploaded_file is not None:
        if uploaded_file.name.lower().endswith("pdf"):
            reader = PyPDF2.PdfReader(uploaded_file)
            content = ""
            for page in reader.pages:
                content += page.extract_text()
        elif uploaded_file.name.lower().endswith("txt"):
            content = uploaded_file.read().decode("utf-8")
        else:
            st.error("Please upload a PDF or TXT file.")
            return None
        return content
    return None

# ---------------------------
# Summarization
# ---------------------------
def summarize(text):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary_result = summarizer(
        "Summarize this document clearly and concisely:\n" + text,
        max_length=150,
        min_length=30,
        do_sample=False
    )
    return summary_result[0]['summary_text']

# ---------------------------
# Sentiment
# ---------------------------
def analyze_sentiment(text):
    sentiment_model = pipeline("sentiment-analysis")
    result = sentiment_model(text[:512])
    return result[0]

# ---------------------------
# NER
# ---------------------------
def find_entities(text):
    ner_model = pipeline("ner", grouped_entities=True)
    entities = ner_model(text[:1000])
    return entities

# ---------------------------
# Topic Extraction (QA)
# ---------------------------
def find_main_topic(text):
    qa_model = pipeline("question-answering")
    question = "What is the main topic of the document?"
    answer = qa_model(question=question, context=text[:1000])
    return answer['answer']

# ---------------------------
# RAG Functions
# ---------------------------
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

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_chunks(chunks):
    embeddings = embedding_model.encode(chunks)
    return embeddings

def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def retrieve_chunks(question, chunks, index, top_k=3):
    q_embedding = embedding_model.encode([question])
    distances, indices = index.search(q_embedding, top_k)
    retrieved = [chunks[i] for i in indices[0]]
    return retrieved

qa_generator = pipeline("text2text-generation", model="facebook/bart-large-cnn")

def rag_answer(question, retrieved_chunks):
    context = " ".join(retrieved_chunks)
    input_text = f"question: {question} context: {context}"
    output = qa_generator(input_text, max_length=150, min_length=30, do_sample=False)
    return output[0]['generated_text']

# ---------------------------
# Fine-tuning Functions
# ---------------------------
def load_or_init_model():
    model_name = "facebook/bart-large-cnn"
    if os.path.exists('./fine_tuned_model'):
        tokenizer = AutoTokenizer.from_pretrained('./fine_tuned_model')
        model = AutoModelForSeq2SeqLM.from_pretrained('./fine_tuned_model')
        return tokenizer, model
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        return tokenizer, model

def fine_tune(text, improved_summary):
    if text and improved_summary.strip() != "":
        tokenizer, model = load_or_init_model()

        input_encodings = tokenizer([text], max_length=1024, truncation=True, padding="max_length", return_tensors="pt")
        target_encodings = tokenizer([improved_summary], max_length=150, truncation=True, padding="max_length", return_tensors="pt")

        class SummaryDataset(torch.utils.data.Dataset):
            def __init__(self, inputs, targets):
                self.inputs = inputs
                self.targets = targets
            def __len__(self):
                return len(self.targets["input_ids"])
            def __getitem__(self, idx):
                item = {key: val[idx] for key, val in self.inputs.items()}
                labels = self.targets["input_ids"][idx]
                labels[labels == tokenizer.pad_token_id] = -100
                item['labels'] = labels
                return item

        dataset = SummaryDataset(input_encodings, target_encodings)

        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=1,
            per_device_train_batch_size=1,
            logging_steps=10,
            save_strategy="no",
            learning_rate=2e-5,
            weight_decay=0.01,
            remove_unused_columns=False,
            report_to=[],
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=tokenizer,
        )

        trainer.train()
        model.save_pretrained('./fine_tuned_model')
        tokenizer.save_pretrained('./fine_tuned_model')
        return "Fine-tuning complete and model saved!"
    return "No improved summary provided; skipping fine-tuning."

# ---------------------------
# Streamlit App
# ---------------------------
def main():
    st.set_page_config(page_title="Document AI Assistant", layout="wide")
    st.title("📄 Document AI Assistant")

    uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])

    if uploaded_file:
        text = get_text(uploaded_file)
        if text:
            st.subheader("📑 Extracted Text")
            with st.expander("Show document text"):
                st.write(text)

            # --- Summary ---
            if st.button("Generate Summary"):
                summary = summarize(text)
                st.subheader("📝 Summary")
                st.write(summary)

                improved = st.text_area("✍️ Improve the summary (optional):", "")
                if st.button("Fine-Tune on Improved Summary"):
                    msg = fine_tune(text, improved)
                    st.success(msg)

            # --- Sentiment ---
            if st.button("Analyze Sentiment"):
                sentiment = analyze_sentiment(text)
                st.subheader("😊 Sentiment")
                st.json(sentiment)

            # --- Entities ---
            if st.button("Find Named Entities"):
                entities = find_entities(text)
                st.subheader("🔍 Named Entities")
                st.json(entities)

            # --- Main Topic ---
            if st.button("Find Main Topic"):
                topic = find_main_topic(text)
                st.subheader("📌 Main Topic")
                st.write(topic)

            # --- RAG Q&A ---
            st.subheader("💬 Ask Questions (RAG)")
            query = st.text_input("Ask a question about the document:")
            if query:
                chunks = chunk_text(text)
                embeddings = embed_chunks(chunks)
                index = build_faiss_index(np.array(embeddings))
                retrieved = retrieve_chunks(query, chunks, index)
                answer = rag_answer(query, retrieved)
                st.write("**Answer:**", answer)

if __name__ == "__main__":
    main()
