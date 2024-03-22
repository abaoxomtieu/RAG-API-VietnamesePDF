from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import GPT4AllEmbeddings
from remove_vector_store import remove_all_vectorstores
pdf_data_path = "data"
vector_db_path = "vectorstores/db_faiss"

def create_db_from_text(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=512, 
        chunk_overlap=50,
        length_function=len
        )
    chunks = text_splitter.split_text(raw_text)
    
    #Embeddings
    embedding_model = GPT4AllEmbeddings(model_file="./models/all-MiniLM-L6-v2-f16.gguf")
    
    #Put into Faiss Vector db_faiss
    db = FAISS.from_texts(texts=chunks, embedding=embedding_model)
    db.save_local(vector_db_path)
    print("Success")
    return db

def create_db_from_files(folder_path='./data'):
    #Load all data in data folder
    loader = DirectoryLoader(folder_path, glob="*.pdf", loader_cls = PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    embedding_model = GPT4AllEmbeddings(model_file="./models/all-MiniLM-L6-v2-f16.gguf")
    db = FAISS.from_documents(chunks, embedding=embedding_model)
    db.save_local(vector_db_path)
    print("Success")
    return db
    
def create_db_from__one_file(pdf_file):
    loader = PyPDFLoader(pdf_file)
    pages = loader.load_and_split(RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50))
    embedding_model = GPT4AllEmbeddings(model_file="./models/all-MiniLM-L6-v2-f16.gguf")
    db = FAISS.from_documents(pages, embedding=embedding_model)
    db.save_local(vector_db_path)
    print("Success")
    return db