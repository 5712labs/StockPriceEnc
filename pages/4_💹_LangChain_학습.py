# pip install langchain=0.0.142
# pip install openai=0.27.4
# pip install tiktoken=0.3.3
# pip install pinecone-client
# https://blog.futuresmart.ai/building-a-document-based-question-answering-system-with-langchain-pinecone-and-llms-like-gpt-4-and-chatgpt#heading-7-embedding-documents-with-openai
import streamlit as st
import os
import tiktoken
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings 
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from langchain.vectorstores import Pinecone

import convert
if convert.check_password() == False:
    st.stop()

loader = DirectoryLoader('./sources', glob='*.txt', loader_cls=TextLoader)
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

os.environ['OPENAI_API_KEY'] = st.secrets["api_key"]
pinecone.init(api_key=f"{st.secrets['api_pine']}", environment='gcp-starter')
st.write('pinecone.list_indexes()')
st.write(pinecone.list_indexes())

st.info(f""" 
## 업로드를 완료하였습니다.
#### Pinecone
[https://app.pinecone.io/](https://app.pinecone.io/).
""")

upsert_button = st.button("Upsert Conversation", key="upsert", type='primary')
if upsert_button:
    # index = pinecone.Index("dwlangchain")
    index_name = 'dwlangchain'
    embedding = OpenAIEmbeddings()
    index = Pinecone.from_documents(texts, embedding, index_name=index_name)
    st.info(f""" 
    ### 업로드를 완료하였습니다.
    #### Pinecone
    [https://app.pinecone.io/](https://app.pinecone.io/).
    """)

# st.write(pinecone.list_indexes())
# st.write(index)

# Upsert sample data (5 8-dimensional vectors)
# index.upsert([
#     ("A", [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
#     ("E", [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
# ])

# index_name = pinecone.Index("dwlangchain")


# import openai

# # get api key from platform.openai.com
# openai.api_key = os.getenv('OPENAI_API_KEY') or 'OPENAI_API_KEY'

# embed_model = "text-embedding-ada-002"
# query = (
#     "Which training method should I use for sentence transformers when " +
#     "I only have pairs of related sentences?"
# )
# res = openai.Embedding.create(
#     input=[query],
#     engine=embed_model
# )

# # retrieve from Pinecone
# xq = res['data'][0]['embedding']
# # get relevant contexts (including the questions)
# res = index.query(xq, top_k=2, include_metadata=True)