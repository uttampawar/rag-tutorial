from uuid import uuid4
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma

# Get ollama embeddings
from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv
import os
import shutil

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

def load_documents():
    file_path = "./data/EA9181_protocol.pdf"
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents

def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    
    docs = []
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents info {len(chunks)} chunks")
    document = chunks[10]
    print(document.page_content)
    print(document.metadata)
    return chunks

def save_to_chroma(chunks: list[Document]):
    # Clear out the database first 
    # NOTE: there is a sqlite3 error so following code is commented out
    # Clear out the database manually by deleting all files in the
    # CHROMA_PATH directory. If you do get sqlite3 error, change 
    # persmission to 777 for CHROMA_PATH directoy only.
    #if os.path.exists(CHROMA_PATH):
    #    shutil.rmtree(CHROMA_PATH)

    uuids = [str(uuid4()) for _ in range(len(chunks))]

    # Create a new DB from the documents
    vector_store.add_documents(documents=chunks, ids=uuids)
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

if __name__ == "__main__":
    main()
