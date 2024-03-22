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

