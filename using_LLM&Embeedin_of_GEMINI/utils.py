from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
import os
import time
from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
vector_db_path = "vectorstores/db_faiss"


def load_llm():
    llm = GoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0,
        verbose=True,
        google_api_key=GOOGLE_API_KEY,
    )
    return llm


def creat_prompt(template):
    prompt = PromptTemplate(template=template, input_variables=[
                            "context", "question"])
    return prompt


def create_qa_chain(prompt, llm, db):
    llm_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(
            search_kwargs={"k": 3}, max_tokens_limit=500),
        return_source_documents=False,
        chain_type_kwargs={'prompt': prompt}

    )
    return llm_chain


def read_vectors_db():
    embedding_model = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001")
    db = FAISS.load_local(vector_db_path, embedding_model,
                          allow_dangerous_deserialization=True)
    return db


db = read_vectors_db()
llm = load_llm()

# Promt
template = """<|im_start|>system\nSử dụng thông tin sau đây để trả lời câu hỏi. Nếu bạn không biết câu trả lời, hãy nói không biết, đừng cố tạo ra câu trả lời\n
    {context}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant"""
prompt = creat_prompt(template)

llm_chain = create_qa_chain(prompt, llm, db)

# Inference
question = "chủ đề paper là gì"
start_time = time.time()
response = llm_chain.invoke({"query": question})
inference_time = time.time() - start_time
print(response)
print("Inference Time:", inference_time, "seconds")
