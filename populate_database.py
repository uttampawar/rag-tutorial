from uuid import uuid4
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma

# Get ollama embeddings
from get_embedding_function import get_embedding_function
from dotenv import load_dotenv
import os
import shutil

CHROMA_PATH = "pdf_chroma-db"
DATA_PATH = "data"

def main():
    generate_data_store()

def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)

def load_documents():
    loader = PyPDFDirectoryLoader(DATA_PATH)
    documents = loader.load()
    return documents

def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )

    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents info {len(chunks)} chunks")
    #document = chunks[10]
    #print(document.page_content)
    #print(document.metadata)
    return chunks

def clean_the_database():
    # Clear out the database first 
    # NOTE: there is a sqlite3 error so following code is commented out
    # Clear out the database manually by deleting all files in the
    # CHROMA_PATH directory. If you do get sqlite3 error, change 
    # persmission to 777 for CHROMA_PATH directoy only.
    #if os.path.exists(CHROMA_PATH):
    #    shutil.rmtree(CHROMA_PATH)
    return

def save_to_chroma(chunks: list[Document]):
    embeddings = get_embedding_function()
    vector_store = Chroma(
        collection_name="example_collection",
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH,
    )

    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or update the documents
    existing_items = vector_store.get(include=[])
    existing_ids = set(existing_items["ids"]) 
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f" -> adding a new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        # Add or create a new DB from the documents
        vector_store.add_documents(new_chunks, ids=new_chunk_ids)
        print(f"Saved {len(new_chunks)} chunks to {CHROMA_PATH}.")
    else:
        print(" No new documents to add") 

def calculate_chunk_ids(chunks):
    # This will create IDs like "data/a.pdf:6.2"
    # page source : page number : chunk index

    last_page_id = None
    current_chunk_index = 0
    for chunk in chunks:
        source = chunk.metadata.get("source") 
        page = chunk.metadata.get("page") 
        current_page_id = f"{source}:{page}"
        # If the page Id is same as the last one, increment the index
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # calculate the chunk ID
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add id to the page metadata of this chunk
        chunk.metadata["id"] = chunk_id

    print(chunks[0].metadata)
    return chunks

if __name__ == "__main__":
    main()
