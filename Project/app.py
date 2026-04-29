import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline

st.title("Chat with Your PDF - RAG Project")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:

    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    docs = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings()

    db = FAISS.from_documents(docs, embeddings)

    question = st.text_input("Ask a question about the PDF")

    if question:

        results = db.similarity_search(question)

        context = results[0].page_content

        generator = pipeline(
            "text-generation",
            model="google/flan-t5-base"
        )

        prompt = f"""
        Answer the question based on the context.

        Context:
        {context}

        Question:
        {question}
        """

        answer = generator(prompt)

        st.write("Answer:")
        st.write(answer[0]["generated_text"])