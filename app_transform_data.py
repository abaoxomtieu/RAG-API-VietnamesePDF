import streamlit as st

from prepare_vector_db import create_db_from_text, create_db_from_files  # Adjust the import path accordingly
vector_db_path = "vectorstores/db_faiss"
from remove_vector_store import remove_all_vectorstores



# Streamlit app
def main():
    st.title("Text and PDF Processor")

    # Text input
    raw_text = st.text_area("Enter Text Here:")
    if st.button("Process Text"):
        if(raw_text != ""):
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
        st.success("Remove database vectors successfully!")
if __name__ == "__main__":
    main()
