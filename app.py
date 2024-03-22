import streamlit as st
import requests
import time
# Import your utility functions
from prepare_vector_db import create_db_from_text, create_db_from_files  # Adjust the import path accordingly
from remove_vector_store import remove_all_vectorstores

# Assuming your FastAPI server is running on this URL
FASTAPI_SERVER_URL = "http://localhost:8000/answer/"

def call_fastapi(question):
    """Send question to FastAPI server and return the answer."""
    response = requests.post(FASTAPI_SERVER_URL, json={"question": question})
    if response.status_code == 200:
        return response.json()["answer"]
    else:
        return "Error: Could not retrieve answer from the server."

def main():
    st.title("AI Powered Text and Question Answering System")

    # Tab configuration
    tab1, tab2 = st.tabs(["Process Text/PDF", "Ask a Question"])

    with tab1:
        st.header("Text and PDF Processor")

        # Text input
        raw_text = st.text_area("Enter Text Here:")
        if st.button("Process Text"):
            if raw_text != "":
                create_db_from_text(raw_text)
                st.success("Text processed successfully!")
            else:
                st.warning("Please enter text")

        # PDF file input
        if st.button("Process PDF from folder data"):
            create_db_from_files()
            st.success("PDF processed successfully!")

        if st.button("Delete all database vectors"):
            remove_all_vectorstores()
            st.success("Removed database vectors successfully!")

    with tab2:
        st.header("Question Answering System")

        # User input for the question
        question = st.text_input("Enter your question:")
        
        if st.button("Ask"):
            if question.strip() != "":
                start_time = time.time()
                answer = call_fastapi(question)
                inference_time = time.time() - start_time
                st.success("Answer: {}".format(answer))
                st.success(inference_time)
            else:
                st.warning("Please enter a question")

if __name__ == "__main__":
    main()
