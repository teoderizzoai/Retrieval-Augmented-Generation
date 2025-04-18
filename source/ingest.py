import os
from typing import List
import nltk
nltk.download('punkt')

from nltk.tokenize import sent_tokenize

def load_documents(folder_path: str) -> List[str]:
    documents = []
    for filename in os.listdir(folder_path):
        with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as f:
            documents.append(f.read())
    return documents

def chunk_text(text: str, max_tokens: int = 300) -> List[str]:
    sentences = sent_tokenize(text)
    chunks, chunk = [], []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence.split())
        if current_length + sentence_length > max_tokens:
            chunks.append(" ".join(chunk))
            chunk = []
            current_length = 0
        chunk.append(sentence)
        current_length += sentence_length

    if chunk:
        chunks.append(" ".join(chunk))
    
    return chunks
