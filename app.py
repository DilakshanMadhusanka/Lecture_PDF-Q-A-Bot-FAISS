import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
import os
import tempfile

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="LECTURE PDF CHATBOT")
st.title("Lecture PDF Q&A Bot")

DATA_PATH = "data"

uploaded_file = st.file_uploader("Upload your Lecture PDF", type="pdf")

if uploaded_file is not None:
    st.success("PDF uploaded successfully!")
    
if uploaded_file is not None:
    pdf_path = os.path.join(DATA_PATH, uploaded_file.name)

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    st.success(f"File saved successfully in data folder: {uploaded_file.name}")

embeddings = OpenAIEmbeddings(
    api_key=openai_api_key
)

vectorstore = FAISS.load_local(
    "vectorstore",
    embeddings,
    allow_dangerous_deserialization=True
)

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

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

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







