!pip install PyPDF2 transformers sentencepiece sentence-transformers faiss-cpu > /dev/null 2>&1

from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from sentence_transformers import SentenceTransformer
import faiss
from google.colab import files
import PyPDF2
import torch
import os
import numpy as np


def getText():
    uploadedFiles = files.upload()
    fileName = list(uploadedFiles.keys())[0]

    if fileName.lower().endswith('pdf'):
        with open(fileName, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            content = ''
            for page in reader.pages:
                content += page.extract_text()
    elif fileName.lower().endswith('txt'):
        content = uploadedFiles[fileName].decode('utf-8')
    else:
        print("Please upload a PDF or TXT file.")
        content = None
    return content


def summarize(text):
    summarizer = pipeline("summarization")
    summaryResult = summarizer("Summarize this document clearly and concisely:\n" + text, max_length=150, min_length=30, do_sample=False)
    print("Summary:")
    print(summaryResult[0]['summary_text'])

    userSummary = input("\nIs this summary good? If not, type your improved summary here. Otherwise, just press Enter:\n")
    return summaryResult[0]['summary_text'], userSummary


def analyzeSentiment(text):
    sentimentModel = pipeline("sentiment-analysis")
    sentimentResult = sentimentModel("What is the overall sentiment of the following text?\n" + text[:512])
    print("Sentiment of the text:")
    print(sentimentResult[0])
    return sentimentResult[0]


def findEntities(text):
    nerModel = pipeline("ner", grouped_entities=True)
    entitiesResult = nerModel("Extract and categorize named entities from the following:\n" + text[:1000])
    print("Named entities found:")
    print(entitiesResult)
    return entitiesResult


def findMainTopic(text):
    qaModel = pipeline("question-answering")
    question = "What is the main topic of the document?"
    answer = qaModel(question=question, context=text[:1000])
    print("Main topic:")
    print(answer['answer'])
    return answer['answer']


# --- RAG Related functions ---

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


def askQuestionsRAG(text):
    chunks = chunk_text(text)
    embeddings = embed_chunks(chunks)
    index = build_faiss_index(np.array(embeddings))

    print("\nYou can now ask questions based on the document. Type 'exit' to quit.\n")
    while True:
        question = input("Ask a question or type 'exit' to quit: ")
        if question.lower() == 'exit':
            break
        retrieved_chunks = retrieve_chunks(question, chunks, index)
        answer = rag_answer(question, retrieved_chunks)
        print("Answer:", answer)


# --- Existing fine tuning and model loading functions ---

def load_or_init_model():
    model_name = "facebook/bart-large-cnn"
    if os.path.exists('./fine_tuned_model'):
        tokenizer = AutoTokenizer.from_pretrained('./fine_tuned_model')
        model = AutoModelForSeq2SeqLM.from_pretrained('./fine_tuned_model')
        print("Loaded fine-tuned model from disk.")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        print("Loaded base model.")
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

        print("\nStarting fine-tuning on your improved summary... This may take some time.")
        trainer.train()
        print("\nFine-tuning complete! Your model has learned from your correction.")
        model.save_pretrained('./fine_tuned_model')
        tokenizer.save_pretrained('./fine_tuned_model')
        print("Saved fine-tuned model and tokenizer to './fine_tuned_model'")
    else:
        print("\nNo improved summary provided; skipping fine-tuning.")


def run():
    state = {}
    state['content'] = getText()
    if state['content']:
        state['summary'], state['userSummary'] = summarize(state['content'])
        state['sentiment'] = analyzeSentiment(state['content'])
        state['entities'] = findEntities(state['content'])
        state['topic'] = findMainTopic(state['content'])
        askQuestionsRAG(state['content'])  # <-- replaced askQuestions with RAG version
        fineTune(state['content'], state['userSummary'])


run()
