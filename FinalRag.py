import os
from langchain_community.document_loaders import PyPDFLoader
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import Chroma
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

def main():
    st.title('RAG PDF Chatbot')

    # Sidebar input for the PDF directory
    st.sidebar.header("PDF Directory")
    pdf_dir = st.sidebar.text_input(
        "Enter the path to your PDF directory:",
        "pdf/"  # Default directory
    )

    # Validate the directory
    if not os.path.isdir(pdf_dir):
        st.error("Please provide a valid directory containing PDF files.")
        return

    # Step 2: Initialize an empty list to store all documents
    all_documents = []

    # Step 3: Iterate over all PDF files in the directory
    st.sidebar.write('Processing PDF files...')
    for file_name in os.listdir(pdf_dir):
        if file_name.endswith(".pdf"):  # Check for PDF files 
            file_path = os.path.join(pdf_dir, file_name)

            # Load PDF content using PyPDFLoader
            loader = PyPDFLoader(file_path)
            documents = loader.load()

            # Append the loaded documents to the main list
            all_documents.extend(documents)
    st.sidebar.write(f"Loaded {len(all_documents)} documents.")


    # Split data into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=500)
    docs = text_splitter.split_documents(all_documents)
    st.sidebar.write(f"Split into {len(docs)} chunks.")

    # Load environment variables
    load_dotenv()

    # Embedding and creating vector store
    st.sidebar.write("Embedding and creating vector store...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    #vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings) 
    vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory="path_to_local_directory")
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

    # Configure the LLM
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3, max_tokens=500)

    # Create the system and user prompt templates
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, say that you don't know. "
        "Use three sentences maximum and keep the answer concise."
        "\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    # Create chains
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # User input
    st.header("Ask a Question")
    user_input = st.text_input("Enter your question:")

    if user_input:
        with st.spinner("Retrieving and generating response..."):
            response = rag_chain.invoke({"input": user_input})
            st.subheader("Answer")
            st.write(response["answer"])

# Ensure the script runs properly
if __name__ == "__main__":
    main()