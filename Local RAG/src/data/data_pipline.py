from loader import download_pdf
from reader import extract_pdf_metadata, read_pdf_pages
from splitter import spacy_sentencize, chunk_sentences, build_chunks
from saver import save_chunks_to_csv
from analytic import plot_chunk_stats
import logging

def process_pdf_pipeline(url, save_path, chunk_size=10, output_csv="chunks_output.csv", show_stats=False):
    download_pdf(url, save_path)
    
    metadata = extract_pdf_metadata(save_path)
    logging.info(f"PDF Title: {metadata.get('title', 'No title')}")

    pages = read_pdf_pages(save_path)
    pages = spacy_sentencize(pages)
    pages = chunk_sentences(pages, chunk_size)
    chunks = build_chunks(pages)
    save_chunks_to_csv(chunks, output_csv)

    if show_stats:
        plot_chunk_stats(chunks)
