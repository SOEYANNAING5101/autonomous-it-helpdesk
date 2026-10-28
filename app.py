
from retrieval.hybrid_searcher import retrieve_documents
from agent.helpdesk_agent import generate_response

def main():
    print("Testing...")
    test_query = "What is the Hybrid card policy? Can you roughly tell me the procedure?"
    print(f"User Query: {test_query}\n")

    print("Retrieving documents: ")
    context = retrieve_documents(test_query)

    if not context:
        print("No documents found")
        return
    print("Generating AI response...")
    rag_output = generate_response(test_query,context)
    print("Answer: \n")
    print(f"Answer: {rag_output.answers}")
    print(f"Cited Chunk id: {rag_output.cited_chunk_ids}")
    print(f"Is Answerable: {rag_output.is_answerable}")

if __name__== '__main__':
    main()