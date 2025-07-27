import fitz
import re
import logging

def text_formatter(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()

def extract_pdf_metadata(pdf_path: str):
    try:
        doc = fitz.open(pdf_path)
        return doc.metadata
    except Exception as e:
        logging.warning(f"Could not extract metadata: {e}")
        return {}

def read_pdf_pages(pdf_path: str):
    doc = fitz.open(pdf_path)
    pages_and_texts = []

    for page_number, page in enumerate(doc, start=1):
        text = page.get_text()
        if not text:
            continue
        formatted = text_formatter(text)
        pages_and_texts.append({
            "page_number": page_number,
            "text": formatted,
            "page_char_count": len(formatted),
            "page_word_count": len(formatted.split()),
            "page_sentence_count_raw": len(re.findall(r'\.', formatted)),
            "page_token_count": len(formatted) / 4,
        })

    return pages_and_texts
