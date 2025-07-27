import gradio as gr
from pathlib import Path
import tempfile

# ✅ Import modular RAG pipeline components (you must implement these)
from rag_pipline import run_data_pipeline, run_embedding, run_inference

# ------------------ Global Configuration ------------------ #
CHUNK_SIZE = 512  # Controls how large each text chunk is (affects retrieval quality)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # HuggingFace model
DEVICE = "cuda"  # Use "cuda" if you have GPU, else switch to "cpu"
TEMP_PDF_PATH = tempfile.gettempdir() + "/uploaded.pdf"  # Temp location to store uploaded PDF

# Global state to keep the vectorstore in memory across interactions
vectorstore_holder = {}

# ------------------ Step 1: Upload + Process PDF ------------------ #
def upload_and_prepare(pdf_file):
    """
    Handles file saving, PDF reading, chunking, and embedding generation.
    """
    pdf_path = Path(TEMP_PDF_PATH)
    pdf_file.save(pdf_path)  # Save the uploaded file

    # 🔹 Run document chunking and preprocessing
    docs = run_data_pipeline(
        pdf_path=pdf_path,
        chunk_size=CHUNK_SIZE,
        show_stats=True  # Optional: prints doc info
    )

    # 🔹 Generate embeddings for vector search
    vectorstore = run_embedding(
        docs=docs,
        model_key=EMBEDDING_MODEL
    )

    return "✅ PDF processed successfully. You can now ask questions.", vectorstore

# ------------------ Step 2: Handle Upload ------------------ #
def handle_upload(pdf_file):
    """
    Wrapper to save the vectorstore globally after processing.
    """
    message, store = upload_and_prepare(pdf_file)
    vectorstore_holder["store"] = store
    return message

# ------------------ Step 3: Handle Inference ------------------ #
def ask_question(query):
    """
    Answers user queries using the embedded knowledge in the uploaded PDF.
    """
    store = vectorstore_holder.get("store")
    if not store:
        return "⚠️ Please upload a PDF first."

    # 🔹 Answer the query using the vectorstore + LLM
    answer = run_inference(query=query, vectorstore=store, device=DEVICE)
    return answer

# ------------------ Gradio UI Definition ------------------ #
with gr.Blocks() as demo:
    gr.Markdown("## 🤖 Ask Your PDF\nUpload a PDF and ask any question based on its content.")

    with gr.Row():
        pdf_input = gr.File(label="📄 Upload PDF", file_types=[".pdf"])
        upload_button = gr.Button("🚀 Upload & Process")

    upload_status = gr.Textbox(label="📢 Status", interactive=False)

    upload_button.click(fn=handle_upload, inputs=[pdf_input], outputs=[upload_status])

    gr.Markdown("---")

    with gr.Row():
        question_input = gr.Textbox(label="❓ Ask a Question")
        ask_button = gr.Button("🔍 Get Answer")
        answer_output = gr.Textbox(label="🧠 Answer", lines=5)

    ask_button.click(fn=ask_question, inputs=[question_input], outputs=[answer_output])

# ------------------ Run the App ------------------ #
if __name__ == "__main__":
    demo.launch()
