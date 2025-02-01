from langchain.document_loaders.pdf import PyPDFDirectoryLoader

DATA_PATH = "data"

def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()

def split_text(documents: list[Document]):
    // TBD
