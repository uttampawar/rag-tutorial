import argparse
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from get_embedding_function import get_embedding_function

CHROMA_PATH = "pdf_chroma-db"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="Input query text")
    args = parser.parse_args()
    input_query_text = args.query_text
    query_rag(input_query_text)

def query_rag(query_text: str):
    embedding_function = get_embedding_function()
    vector_store = Chroma(
        collection_name="example_collection",
        embedding_function=embedding_function,
        persist_directory=CHROMA_PATH,
    )

    # Search the DB
    results = vector_store.similarity_search_with_score(query_text, k=5)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)

    model = Ollama(model="llama3.2") # Replace with any ollama supported models such as "mistral" or "llama3.2" 
    response_text = model.invoke(prompt)
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}" 
    print(formatted_response)

if __name__ == "__main__":
    main()
