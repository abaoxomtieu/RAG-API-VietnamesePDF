from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import FAISS
from accelerate import Accelerator
import os
import time
os.environ['KMP_DUPLICATE_LIB_OK']='True'
# Cau hinh
accelerator = Accelerator()
model_file = "models/vinallama-7b-chat_q5_0.gguf"
vector_db_path = "vectorstores/db_faiss"

#LOAD WITH GPU

def load_llm(model_file):
    config = {'max_new_tokens': 500, 'repetition_penalty': 1.1, 'context_length': 512, 'temperature':0.1, 'gpu_layers':30}
    llm = CTransformers(
        model=model_file,
        model_type="llama",
        config=config
    )
    llm, config = accelerator.prepare(llm, config)
    return llm

# #LOAD WITH CPU
# def load_llm(model_file):
#     llm = CTransformers(
#         model=model_file,
#         model_type="llama",
#         max_new_tokens=500,
#         temperature=0.01
#     )
#     return llm

def creat_prompt(template):
    prompt = PromptTemplate(template = template, input_variables=["context", "question"])
    return prompt


def create_qa_chain(prompt, llm, db):
    llm_chain = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type= "stuff",
        retriever = db.as_retriever(search_kwargs = {"k":3}, max_tokens_limit=500),
        return_source_documents = False,
        chain_type_kwargs= {'prompt': prompt}

    )
    return llm_chain

def read_vectors_db():
    embedding_model = GPT4AllEmbeddings(model_file="models/all-MiniLM-L6-v2-f16.gguf")
    db = FAISS.load_local(vector_db_path, embedding_model,allow_dangerous_deserialization=True)
    return db


db = read_vectors_db()
llm = load_llm(model_file)

#Promt
template = """<|im_start|>system\nSử dụng thông tin sau đây để trả lời câu hỏi. Nếu bạn không biết câu trả lời, hãy nói không biết, đừng cố tạo ra câu trả lời\n
    {context}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant"""
prompt = creat_prompt(template)

llm_chain  =create_qa_chain(prompt, llm, db)

#Inference
question = "chủ đề paper là gì"
start_time = time.time()
response = llm_chain.invoke({"query": question})
inference_time = time.time() - start_time
print(response)
print("Inference Time:", inference_time, "seconds")