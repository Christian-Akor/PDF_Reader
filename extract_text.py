from PyPDF2 import PdfReader
from pytesseract import image_to_string
from PIL import Image

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_image(image_path):
    return image_to_string(Image.open(image_path))

pdf_text = extract_text_from_pdf("document.pdf")
print(pdf_text[:500])  # Show first 500 characters


def chunk_text(text, chunk_size=500):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# Example Usage
chunks = chunk_text(pdf_text)
print(chunks[:2])  # Show first 2 chunks


from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_embeddings(chunks):
    return [model.encode(chunk) for chunk in chunks]

# Example Usage
embeddings = generate_embeddings(chunks)


import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def find_relevant_chunks(question, chunks, embeddings):
    question_embedding = model.encode(question)
    similarities = cosine_similarity([question_embedding], embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:2]  # Top 2 most similar chunks
    return [chunks[i] for i in top_indices]

# Example Usage
question = "What is the purpose of this document?"
relevant_chunks = find_relevant_chunks(question, chunks, embeddings)
print(relevant_chunks)


from transformers import pipeline

qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

def get_answer(question, context):
    return qa_pipeline(question=question, context=context)['answer']

# Example Usage
context = " ".join(relevant_chunks)
answer = get_answer(question, context)
print("Answer:", answer)


import os

def extract_text_from_pdf(pdf_path):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"The file '{pdf_path}' was not found.")
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text
