from spacy.lang.en import English
from tqdm.auto import tqdm
import re
import pandas as pd
import logging

def spacy_sentencize(pages_and_texts):
    nlp = English()
    nlp.add_pipe("sentencizer")

    for item in tqdm(pages_and_texts, desc="Sentence splitting"):
        doc = nlp(item["text"])
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        item["sentences"] = sentences
        item["page_sentence_count_spacy"] = len(sentences)

    return pages_and_texts

def chunk_sentences(pages_and_texts, chunk_size=10):
    for item in tqdm(pages_and_texts, desc="Chunking sentences"):
        sentences = item["sentences"]
        item["sentence_chunks"] = [sentences[i:i+chunk_size] for i in range(0, len(sentences), chunk_size)]
        item["num_chunks"] = len(item["sentence_chunks"])
    return pages_and_texts

def build_chunks(pages_and_texts):
    all_chunks = []

    for item in pages_and_texts:
        for chunk in item["sentence_chunks"]:
            joined = " ".join(chunk)
            joined = re.sub(r'\.([A-Z])', r'. \1', joined)
            all_chunks.append({
                "page_number": item["page_number"],
                "chunk_text": joined.strip(),
                "chunk_char_count": len(joined),
                "chunk_word_count": len(joined.split()),
                "chunk_token_estimate": len(joined) / 4
            })

    return all_chunks


def save_chunks_to_csv(chunks, output_path="chunks_output.csv"):
    df = pd.DataFrame(chunks)
    df.to_csv(output_path, index=False)
    logging.info(f"✅ Chunks saved to {output_path}")
