from dotenv import load_dotenv
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from remove_vector_store import remove_all_vectorstores
pdf_data_path = "data"
vector_db_path = "vectorstores/db_faiss"
load_dotenv()
print(os.getenv("GOOGLE_API_KEY"))


def create_db_from_text(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=512,
        chunk_overlap=50,
        length_function=len
    )
    chunks = text_splitter.split_text(raw_text)
    embedding_model = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001"
    )
    # Put into Faiss Vector db_faiss
    db = FAISS.from_texts(texts=chunks, embedding=embedding_model)
    db.save_local(vector_db_path)
    print("Success")
    return db


def create_db_from_files(folder_path='./data'):
    # Load all data in data folder
    loader = DirectoryLoader(folder_path, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    embedding_model = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001"
    )
    db = FAISS.from_documents(chunks, embedding=embedding_model)
    db.save_local(vector_db_path)
    print("Success")
    return db


def create_db_from__one_file(pdf_file):
    loader = PyPDFLoader(pdf_file)
    pages = loader.load_and_split(
        RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50))
    embedding_model = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001"
    )
    db = FAISS.from_documents(pages, embedding=embedding_model)
    db.save_local(vector_db_path)
    print("Success")
    return db


create_db_from_files()


# transfrom text into vectorDB
# raw_text = """ Transformer attention thông thường thực hiện attention trên toàn bộ
# feature map nó dẫn đến độ phức tạp của thuật toán tăng cao khi spatial size của feature map tăng.
# Tác giả đưa ra một kiểu attention mới mà chỉ attend
# vào một số sample locations (sample locations này cũng không cố định mà được học trong
# quá trình training tương tự như trong deformable convolution)
# qua đó giúp giảm độ phức tạp của thuật toán và làm giảm thời gian training mô hình. """
# create_db_from_text(raw_text)


# transfrom pdf file into vectorDB
# pdf_file = "./data//Le Van Hao Phuong phap NCKH-XHNV-2015.pdf"
# create_db_from__one_file(pdf_file)


# transfrom folder contain pdf file into vectorDB
# create_db_from_files()


# Remove all vectors DB
# remove_all_vectorstores()
