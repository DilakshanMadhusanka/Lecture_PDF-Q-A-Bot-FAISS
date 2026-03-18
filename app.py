import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from ingest import index_pdf

load_dotenv()

openai_api_key = st.secrets.get("OPENAI_API_KEY")

st.set_page_config(page_title="LECTURE PDF CHATBOT")
st.title("Lecture PDF Q&A Bot")

DATA_PATH = "data"
VECTOR_PATH = "vectorstore"

os.makedirs(DATA_PATH, exist_ok=True)

uploaded_file = st.file_uploader("Upload your Lecture PDF", type="pdf")

if uploaded_file:

    pdf_path = os.path.join(DATA_PATH, uploaded_file.name)

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    index_pdf(pdf_path)

if os.path.exists(VECTOR_PATH):

    embeddings = OpenAIEmbeddings(api_key=openai_api_key)

    vectorstore = FAISS.load_local(
        VECTOR_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    prompt = ChatPromptTemplate.from_template(
        """Answer the question using ONLY the context below.

        Context:
        {context}

        Question: {question}
        """
    )

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
    )

    query = st.text_input("Ask a question from your lecture PDFs:")

    if query:
        with st.spinner("Thinking..."):
            answer = rag_chain.invoke(query)
            st.success(answer.content)
