import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from sentence_transformers import SentenceTransformer
import faiss
import PyPDF2
import torch
import os
import numpy as np

# Load models once with caching
@st.cache_resource
def get_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def get_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

@st.cache_resource
def get_sentiment_model():
    return pipeline("sentiment-analysis")

@st.cache_resource
def get_ner_model():
    return pipeline("ner", grouped_entities=True)

@st.cache_resource
def get_qa_model():
    return pipeline("question-answering")

@st.cache_resource
def get_qa_generator():
    return pipeline("text2text-generation", model="facebook/bart-large-cnn")

@st.cache_resource
def get_tokenizer_and_model():
    model_path = './fine_tuned_model'
    if os.path.exists(model_path):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        st.success("Loaded fine-tuned model.")
    else:
        tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
        model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
    return tokenizer, model


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


def summarize(text):
    summarizer = get_summarizer()
    summaryResult = summarizer("Summarize this document clearly and concisely:\n" + text, max_length=150, min_length=30, do_sample=False)
    st.write("### Summary:")
    st.write(summaryResult[0]['summary_text'])

    userSummary = st.text_input("Is this summary good? If not, type your improved summary here. Otherwise, leave blank:")
    return summaryResult[0]['summary_text'], userSummary


def analyzeSentiment(text):
    sentimentModel = get_sentiment_model()
    sentimentResult = sentimentModel("What is the overall sentiment of the following text?\n" + text[:512])
    st.write("### Sentiment of the text:")
    st.write(sentimentResult[0])
    return sentimentResult[0]


def findEntities(text):
    nerModel = get_ner_model()
    entitiesResult = nerModel("Extract and categorize named entities from the following:\n" + text[:1000])
    st.write("### Named entities found:")
    st.write(entitiesResult)
    return entitiesResult


def findMainTopic(text):
    qaModel = get_qa_model()
    question = "What is the main topic of the document?"
    answer = qaModel(question=question, context=text[:1000])
    st.write("### Main topic:")
    st.write(answer['answer'])
    return answer['answer']


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


def embed_chunks(chunks):
    embedding_model = get_embedding_model()
    embeddings = embedding_model.encode(chunks)
    return embeddings


def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index


def retrieve_chunks(question, chunks, index, top_k=3):
    question_embedding = get_embedding_model().encode([question])
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
    index = build_faiss_index(np.array(embeddings))

    st.write("\nYou can now ask questions based on the document. Type your question and press enter.")

    question = st.text_input("Ask a question or leave blank to stop:", key="rag_question")
    if question:
        try:
            retrieved_chunks = retrieve_chunks(question, chunks, index)
            answer = rag_answer(question, retrieved_chunks)
            st.write("**Answer:**", answer)
        except Exception as e:
            st.error(f"Error retrieving answer: {e}")


def fineTune(text, improvedSummary):
    if text and improvedSummary.strip() != "":
        tokenizer, model = get_tokenizer_and_model()

        inputEncodings = tokenizer([text], max_length=1024, truncation=True, padding="max_length", return_tensors="pt")
        targetEncodings = tokenizer([improvedSummary], max_length=150, truncation=True, padding="max_length", return_tensors="pt")

        class SummaryDataset(torch.utils.data.Dataset):
            def __init__(self, inputs, targets):
                self.inputs = inputs
                self.targets = targets
            def __len__(self):
                return self.targets["input_ids"].size(0)
            def __getitem__(self, idx):
                item = {key: val[idx] for key, val in self.inputs.items()}
                labels = self.targets["input_ids"][idx]
                labels[labels == tokenizer.pad_token_id] = -100
                item['labels'] = labels
                return item

        dataset = SummaryDataset(inputEncodings, targetEncodings)

        trainingArgs = TrainingArguments(
            output_dir='./results',
            num_train_epochs=1,
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

        st.write("\nStarting fine-tuning on your improved summary... This may take some time.")
        trainer.train()
        st.write("\nFine-tuning complete! Your model has learned from your correction.")
        model.save_pretrained('./fine_tuned_model')
        tokenizer.save_pretrained('./fine_tuned_model')
        st.write("Saved fine-tuned model and tokenizer to './fine_tuned_model'")
    else:
        st.write("\nNo improved summary provided; skipping fine-tuning.")


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
