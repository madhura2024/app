import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from sentence_transformers import SentenceTransformer
import faiss
import PyPDF2
import torch
import os
import numpy as np
import time

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
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    summaryResult = summarizer(text, max_length=150, min_length=30, do_sample=False)

    st.write("### Summary:")
    st.write(summaryResult[0]['summary_text'])

    userSummary = st.text_input("Is this summary good? If not, type your improved summary here. Otherwise, leave blank:")
    return summaryResult[0]['summary_text'], userSummary


# ------------------ Sentiment Analysis ------------------
def analyzeSentiment(text):
    sentimentModel = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    sentimentResult = sentimentModel(text[:512])
    st.write("### Sentiment of the text:")
    st.write(sentimentResult[0])
    return sentimentResult[0]


# ------------------ NER ------------------
def findEntities(text):
    nerModel = pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True)
    entitiesResult = nerModel(text[:1000])
    st.write("### Named entities found:")
    st.write(entitiesResult)
    return entitiesResult


# ------------------ Main Topic ------------------
def findMainTopic(text):
    qaModel = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
    question = "What is the main topic of the document?"
    answer = qaModel(question=question, context=text[:1000])
    st.write("### Main topic:")
    st.write(answer['answer'])
    return answer['answer']


# ------------------ RAG Helpers ------------------
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
    question_embedding = embedding_model.encode([question])
    distances, indices = index.search(question_embedding, top_k)
    retrieved_chunks = [chunks[i] for i in indices[0]]
    return retrieved_chunks


qa_generator = pipeline("text2text-generation", model="facebook/bart-large-cnn")

def rag_answer(question, retrieved_chunks):
    context = " ".join(retrieved_chunks)
    input_text = f"question: {question} context: {context}"
    output = qa_generator(input_text, max_length=150, min_length=30, do_sample=False)
    return output[0]['generated_text']


# ------------------ RAG Function with One Question ------------------
def askQuestionsRAG(text):
    with st.spinner("Processing your question..."):
        try:
            # Step 1: Chunk the text into smaller chunks for embedding and retrieval
            chunks = chunk_text(text)
            st.write(f"Total number of chunks: {len(chunks)}")  # Debug: check the number of chunks
            if len(chunks) == 0:
                st.error("No chunks created. Please check the document content.")
                return

            # Step 2: Create embeddings for the chunks
            embeddings = embed_chunks(chunks)
            st.write(f"Sample embeddings: {embeddings[:2]}")  # Debug: show sample embeddings
            if len(embeddings) == 0:
                st.error("Failed to generate embeddings. Please check the chunking process.")
                return

            # Step 3: Build the FAISS index for efficient retrieval
            index = build_faiss_index(np.array(embeddings))
            st.write("FAISS index created successfully.")  # Debug: check if FAISS index is created

            # Step 4: Ask the user for a question
            question = st.text_input("Ask a question about the document:")

            if question:
                st.write(f"Received question: {question}")  # Debug: log the received question

                # Step 5: Retrieve relevant chunks from the FAISS index
                retrieved_chunks = retrieve_chunks(question, chunks, index)
                st.write(f"Retrieved chunks: {retrieved_chunks[:3]}")  # Debug: show top 3 retrieved chunks

                if len(retrieved_chunks) == 0:
                    st.error("No relevant chunks found. Try asking a different question.")
                    return

                # Step 6: Generate an answer using the retrieved chunks
                answer = rag_answer(question, retrieved_chunks)
                st.write("**Answer:**", answer)
        except Exception as e:
            st.error(f"An error occurred during the question-answering process: {str(e)}")


# ------------------ Fine-Tuning ------------------
def load_or_init_model():
    model_name = "facebook/bart-large-cnn"
    if os.path.exists('./fine_tuned_model'):
        tokenizer = AutoTokenizer.from_pretrained('./fine_tuned_model')
        model = AutoModelForSeq2SeqLM.from_pretrained('./fine_tuned_model')
        st.write("Loaded fine-tuned model from disk.")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        st.write("Loaded base model.")
    return tokenizer, model


def fineTune(text, improvedSummary):
    if text and improvedSummary.strip() != "":
        tokenizer, model = load_or_init_model()

        inputEncodings = tokenizer([text], max_length=1024, truncation=True, padding="max_length", return_tensors="pt")
        targetEncodings = tokenizer([improvedSummary], max_length=150, truncation=True, padding="max_length", return_tensors="pt")

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

        dataset = SummaryDataset(inputEncodings, targetEncodings)

        trainingArgs = TrainingArguments(
            output_dir='./results',
            num_train_epochs=3,
            per_device_train_batch_size=1,
            logging_steps=10,
            save_strategy="no",
            learning_rate=2e-5,
            weight_decay=0.01,
            save_total_limit=1,
            remove_unused_columns=False,
            report_to=[],
        )

        trainer = Trainer(
            model=model,
            args=trainingArgs,
            train_dataset=dataset,
            tokenizer=tokenizer,
        )

        st.write("Starting fine-tuning on your improved summary... This may take some time.")
        trainer.train()
        st.write("Fine-tuning complete! Your model has learned from your correction.")
        model.save_pretrained('./fine_tuned_model')
        tokenizer.save_pretrained('./fine_tuned_model')
        st.write("Saved fine-tuned model and tokenizer to './fine_tuned_model'")
    else:
        st.write("No improved summary provided; skipping fine-tuning.")


# ------------------ Main Run ------------------
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
