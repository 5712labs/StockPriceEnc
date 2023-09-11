# pip install langchain=0.0.142
# pip install openai=0.27.4
# pip install tiktoken=0.3.3
# pip install chromadb=0.3.21
# https://anpigon.tistory.com/389
# https://rading.kr/programming/machine-learn/6563/
# https://www.youtube.com/watch?v=ftXLn9DE7YY
import streamlit as st
import os
# import platform
# import openai
# import chromadb
# import langchain
import tiktoken
from langchain.vectorstores import Chroma 
from langchain.embeddings.openai import OpenAIEmbeddings 
from langchain.text_splitter import RecursiveCharacterTextSplitter 
# from langchain.chat_models import ChatOpenAI
# from langchain.chains import ChatVectorDBChain 
# from langchain.document_loaders import GutenbergLoader
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader

# loader = GutenbergLoader("https://www.gutenberg.org/cache/epub/69972/pg69972.txt")
loader = DirectoryLoader('./sources', glob='*.txt', loader_cls=TextLoader)
# loader = DirectoryLoader('./sources', glob='dtms_01.txt', loader_cls=TextLoader)
documents = loader.load()

def num_tokens_from_string(string: str, encoding_name: str) -> int:  
    """Returns the number of tokens in a text string."""  
    encoding = tiktoken.get_encoding(encoding_name)  
    num_tokens = len(encoding.encode(string))  
    return num_tokens

# 텍스트를 청크(chunk) 단위로 분할하기
chunk_size = 1000
text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=20)
texts = text_splitter.split_documents(documents)
st.write(f'{len(documents)}개의 문서를 {chunk_size} 청크 단위로 {len(texts)}개의 문서로 분할 하였습니다.')
st.write(texts)
# persist_directory="/content/drive/My Drive/Colab Notebooks/chroma/romeo"

st.stop()


persist_directory="db"
os.environ['OPENAI_API_KEY'] = st.secrets["api_key"]
embedding = OpenAIEmbeddings()
vectordb = Chroma.from_documents(
    documents=texts,
    embedding=embedding, 
    persist_directory=persist_directory)  
vectordb.persist()
vectordb = None



# 불러오기
vectordb = Chroma(
    embedding_function=embedding, 
    persist_directory=persist_directory)  
retriever = vectordb.as_retriever()
# retriever = vectordb.as_retriever(search_kwargs={"k": 3})
docs = retriever.get_relevant_documents("퇴사자 수 알려줘")

for doc in docs:
    st.write(doc.metadata["source"])

# model = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
# chain = ChatVectorDBChain.from_llm(model, vectordb, return_source_documents=True)

# query = "산업은행의 중도 퇴사자 수는?"
# result = chain({"question": query, "chat_history": []})
# st.write(result)



