from uuid import uuid4
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma

# Get ollama embeddings
from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv
import os
import shutil

DATA_PATH = "data/books"
CHROMA_PATH = "chroma-db"

embeddings = OllamaEmbeddings(model="llama3.2")

vector_store = Chroma(
               collection_name="example_collection",
               embedding_function=embeddings,
               persist_directory=CHROMA_PATH,
             )

def main():
    generate_data_store()

def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)
    #vector = embeddings.embed_query("apple")
    #print(vector)
    #print(len(vector))

def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.md")
    documents = loader.load()
    print(documents[0].page_content[:100])
    return documents

def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    
    docs = []
    chunks = text_splitter.split_documents(documents)
    for cidx in range(len(chunks)):
        content = chunks[cidx].page_content
        metadata = chunks[cidx].metadata
        print(cidx)
        print(content)
        print(metadata)
        page = Document(page_content=content, metadata=metadata)
        docs.append(page)
        print(f"-----------------------------------")

    print(f"Split {len(documents)} documents info {len(chunks)} chunks")
    #document = chunks[10]
    #print(document.page_content)
    #print(document.metadata)
    return chunks

def save_to_chroma(chunks: list[Document]):
    # Clear out the database first
    #if os.path.exists(CHROMA_PATH):
    #    shutil.rmtree(CHROMA_PATH)
    uuids = [str(uuid4()) for _ in range(len(chunks))]
    #print(uuids)
    # Create a new DB from the documents
    vector_store.add_documents(documents=chunks, ids=uuids)
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

if __name__ == "__main__":
    main()
