from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
# from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import FAISS
import os
import time
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
load_dotenv()
app = FastAPI()

# Load LLM model
vector_db_path = "vectorstores/db_faiss"

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
print(GOOGLE_API_KEY)
llm = GoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    verbose=True,
    google_api_key=GOOGLE_API_KEY,
)


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
        model="models/embedding-001"
    )
    db = FAISS.load_local(vector_db_path, embedding_model,
                          allow_dangerous_deserialization=True)
    return db


# Load models and create chain
db = read_vectors_db()
template = """<|im_start|>system\nSử dụng thông tin sau đây để trả lời câu hỏi. Nếu bạn không biết câu trả lời, hãy nói không biết, đừng cố tạo ra câu trả lời\n
    {context}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant"""
prompt = creat_prompt(template)
llm_chain = create_qa_chain(prompt, llm, db)


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
