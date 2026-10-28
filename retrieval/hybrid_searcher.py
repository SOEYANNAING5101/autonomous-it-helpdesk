import os
import re
from dotenv import load_dotenv
from openai import OpenAI
import numpy as np
from ingestion.database import conn, bm25index,documents_dict,corpus_ids
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
def expand_query(user_query):
    system_instruction = """
    You are an IT Helpdesk search optimizer for IBM Security Access Manager for Enterprise Single Sign-On, Help Desk Guide.
    Expand the user's problem using ONLY relevant technical synonyms, root causes, 
    and related concepts. Do not answer the question. 
    Output a comma-separated list of keywords.
    """
    response = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[
            {"role":"developer","content":system_instruction},
            {"role":"user","content":user_query}
        ]
    )
    llm_expansion = response.choices[0].message.content.strip()
    return f"{user_query} {llm_expansion}"

def compute_rrf(vector_ranks,bm25_ranks,k=60):
    rrf_scores = {}
    for rank,doc_id in enumerate(vector_ranks):
        if doc_id not in rrf_scores:
            rrf_scores[doc_id] = 0.0
        rrf_scores[doc_id] += 1/(k+rank)
    for rank,doc_id in enumerate(bm25_ranks):
        if doc_id not in rrf_scores:
            rrf_scores[doc_id] = 0.0
        rrf_scores[doc_id] += 1/(k+rank)
    sorted_result = sorted(rrf_scores.items(),key=lambda item: item[1],reverse=True)
    return [item[0] for item in sorted_result]

def retrieve_documents(user_query,top_k=3):
    expanded_query = expand_query(user_query)
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=expanded_query
    )
    user_vector = np.array(response.data[0].embedding)

    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id 
        FROM documents 
        ORDER BY embedding <=> %s::vector
        LIMIT 10
        """,
        (user_vector,)
    )
    vector_leaderboard = [row[0] for row in cursor.fetchall()]
    cursor.close()

    # Top 10 BM25 keyword matches
    tokenized_query = re.findall(f'w+',expanded_query.lower())
    raw_bm25_score = bm25index.get_scores(tokenized_query)

    scored_bm25_ids = list(zip(corpus_ids,raw_bm25_score))
    scored_bm25_ids.sort(key=lambda x: x[1],reverse=True)
    bm25_leaderboard = [doc_id for doc_id,score in scored_bm25_ids if score>0]

    final_ranked_ids = compute_rrf(vector_leaderboard,bm25_leaderboard)
    winning_ids = final_ranked_ids[:top_k]
    if not winning_ids:
        return

    context_blocks = []
    for doc_id in winning_ids:
        doc_data = documents_dict[doc_id]
        context_text = f"Document: {doc_id} \n Section: {doc_data['section']} \n Content: {doc_data['content']}"
        context_blocks.append(context_text)
    full_context = "\n\n".join(context_blocks)
    return full_context