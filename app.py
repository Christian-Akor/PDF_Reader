from flask import Flask, request, jsonify
from extract_text import extract_text_from_pdf, chunk_text, generate_embeddings, find_relevant_chunks, get_answer

app = Flask(__name__)

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.json
    question = data['question']
    pdf_path = data['pdf_path']

    # Extract text, chunk it, and generate embeddings
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)
    embeddings = generate_embeddings(chunks)

    # Find relevant chunks and generate answer
    relevant_chunks = find_relevant_chunks(question, chunks, embeddings)
    context = " ".join(relevant_chunks)
    answer = get_answer(question, context)

    return jsonify({"answer": answer})

if __name__ == '__main__':
    app.run(debug=True)
