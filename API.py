from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import FAISS
from accelerate import Accelerator
import os
import time

app = FastAPI()

# Configure Accelerator
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
accelerator = Accelerator()

# Load LLM model
model_file = "models/vinallama-7b-chat_q5_0.gguf"
vector_db_path = "vectorstores/db_faiss"

def load_llm(model_file):
    config = {'max_new_tokens': 500, 'repetition_penalty': 1.1, 'context_length': 512, 'temperature':0.1, 'gpu_layers':50}
    llm = CTransformers(
        model=model_file,
        model_type="llama",
        config=config
    )
    llm, config = accelerator.prepare(llm, config)
    return llm

def creat_prompt(template):
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    return prompt

def create_qa_chain(prompt, llm, db):
    llm_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 3}, max_tokens_limit=500),
        return_source_documents=False,
        chain_type_kwargs={'prompt': prompt}
    )
    return llm_chain

def read_vectors_db():
    embedding_model = GPT4AllEmbeddings(model_file="models/all-MiniLM-L6-v2-f16.gguf")
    db = FAISS.load_local(vector_db_path, embedding_model, allow_dangerous_deserialization=True)
    return db

# Load models and create chain
db = read_vectors_db()
llm = load_llm(model_file)
template = """system\nSử dụng thông tin sau đây để trả lời câu hỏi. Nếu bạn không biết câu trả lời, hãy nói không biết, đừng cố tạo ra câu trả lời\n
    {context}\nuser\n{question}\nassistant"""
prompt = creat_prompt(template)
llm_chain = create_qa_chain(prompt, llm, db)

# Define request body model
class QuestionRequest(BaseModel):
    question: str

@app.post("/answer/")
def answer_question(question_request: QuestionRequest):
    try:
        
        start_time = time.time()
        response = llm_chain.invoke({"query": question_request.question})
        inference_time = time.time() - start_time
        print(response)
        print("Inference Time:", inference_time, "seconds")
        return {"answer": response["result"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the FastAPI application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
