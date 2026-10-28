import os
import re
import psycopg2
from rank_bm25 import BM25Okapi
from pgvector.psycopg2 import register_vector
from dotenv import load_dotenv

load_dotenv()

print("Connecting to PostgreSQL...")
conn = psycopg2.connect(os.getenv("DATABASE_URL"))
register_vector(conn)

def load_bm25_index():
    print("Loading documents from database for BM25 indexing...")
    cursor = conn.cursor()
    cursor.execute("SELECT id,content,metadata->>'section' FROM documents")
    rows = cursor.fetchall()
    cursor.close()

    documents_dict = {}
    tokenized_corpus = []
    corpus_ids = []
    for row in rows:
        doc_id = row[0]
        content = row[1]
        section = row[2]

        documents_dict[doc_id] = {"content":content,"section":section}
        corpus_ids.append(doc_id)

        tokenized_corpus.append(re.findall(r'\w+',content.lower()))
    bm25 = BM25Okapi(tokenized_corpus)
    print(f"BM25 initialized with {len(corpus_ids)} documents.\n")
    return bm25,documents_dict,corpus_ids

bm25index,documents_dict,corpus_ids = load_bm25_index()