# pip install langchain=0.0.142
# pip install openai=0.27.4
# pip install tiktoken=0.3.3
# pip install chromadb=0.3.21
# https://anpigon.tistory.com/389
import streamlit as st
import os
import platform
import openai
import chromadb
import langchain
import tiktoken

from langchain.vectorstores import Chroma 
from langchain.embeddings.openai import OpenAIEmbeddings 
from langchain.text_splitter import RecursiveCharacterTextSplitter 
# from langchain.chains import ChatVectorDBChain 
# from langchain.llms import OpenAI
from langchain.document_loaders import GutenbergLoader
# https://www.youtube.com/watch?v=ftXLn9DE7YY
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader

import pinecone
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

import convert
if convert.check_password() == False:
    st.stop()
 
# index = pinecone.Index("dwlangchain")
index_name = 'dwlangchain'
os.environ['OPENAI_API_KEY'] = st.secrets["api_dw"]

embed = OpenAIEmbeddings(
    # model='text-embedding-ada-002',
    # openai_api_key=st.secrets["api_dw"]
)

text_field = "text"
index = pinecone.Index(index_name)
vectorstore = Pinecone(
    index, embed.embed_query, text_field
)

# query = "어학성적"
query = "모집인원"


# xq = Pinecone.from_existing_index(index_name, embed)
# res = index.query(xq, top_k=5, include_metadata=True)
# st.write(res)

# similar_docs = vectorstore.similarity_search_with_score(
#     query,  # our search query
#     k=2
# )
# st.write('vectorstore')
# st.write(similar_docs)

# query = "어학 점수는 어떻게 되나요"
# embedding = OpenAIEmbeddings()
# testingIndex = Pinecone.from_existing_index(index_name, embedding)
# similar_docs = testingIndex.similarity_search_with_score(
# # similar_docs = testingIndex.similarity_search(
#     query,  # our search query
#     k=2
# )
# st.write('from_existing_index')
# st.write(similar_docs)

# chat completion llm
llm = ChatOpenAI(
    openai_api_key=st.secrets["api_dw"],
    model_name='gpt-3.5-turbo',
    temperature=0.0
)
# conversational memory
conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=5,
    return_messages=True
)
# retrieval qa chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(top_k = 1)
)
result = qa.run(query)
st.write(result)

qa_with_sources = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)
result = qa_with_sources(query)
st.write(result)

from langchain.agents import Tool
tools = [
    Tool(
        name='Knowledge Base',
        func=qa.run,
        description=(
            'use this tool when answering general knowledge queries to get '
            'more information about the topic'
        )
    )
]
from langchain.agents import initialize_agent

agent = initialize_agent(
    agent='chat-conversational-react-description',
    tools=tools,
    llm=llm,
    verbose=True,
    max_iterations=3,
    early_stopping_method='generate',
    memory=conversational_memory
)
hist = agent(query)
st.write(hist)

st.stop()

llm = ChatOpenAI(
    openai_api_key=st.secrets["api_dw"],
    model_name='gpt-3.5-turbo',
    temperature=0.0
)
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=testingIndex.as_retriever()
)
qa.run(query)

st.write(qa)



st.stop()
from langchain.chains.question_answering import load_qa_chain
# llm = ChatOpenAI(
#     temperature = 0.0
# )
# model_name = "text-davinci-003"
model_name = "gpt-3.5-turbo"
# model_name = "gpt-4"
llm = OpenAI(model_name = model_name)
chain = load_qa_chain(llm, chain_type="stuff")
answer = chain.run(input_documents=similar_docs, question=query)

st.write(answer)


st.stop()

# query_response = index.query(
#     namespace="dwlangchain",
#     top_k=10,
#     vector=[0.1, 0.2, 0.3, 0.4],
#     sparse_vector={
#         'indices': [3],
#         'values':  [0.8]
#     }
# )
# st.write(answer)

# def get_similiar_docs(query, k=2, score=False):
#   if score:
#     similar_docs = index.similarity_search_with_score(query, k=k)
#   else:
#     similar_docs = index.similarity_search(query, k=k)
#   return similar_docs


# loader = GutenbergLoader("https://www.gutenberg.org/cache/epub/69972/pg69972.txt")
# loader = DirectoryLoader('./sources', glob='*.txt', loader_cls=TextLoader)
# documents = loader.load()

# def num_tokens_from_string(string: str, encoding_name: str) -> int:  
#     """Returns the number of tokens in a text string."""  
#     encoding = tiktoken.get_encoding(encoding_name)  
#     num_tokens = len(encoding.encode(string))  
#     return num_tokens

# # 텍스트를 청크(chunk) 단위로 분할하기
# chunk_size = 10
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
# texts = text_splitter.split_documents(documents)
# st.write(f'{len(documents)}개의 문서를 {chunk_size} 청크 단위로 {len(texts)}개의 문서로 분할 하였습니다.')
# st.write(texts)
# # persist_directory="/content/drive/My Drive/Colab Notebooks/chroma/romeo"

persist_directory="db"
os.environ['OPENAI_API_KEY'] = st.secrets["api_key"]
embedding = OpenAIEmbeddings()  
# vectordb = Chroma.from_documents(
#     documents=texts,
#     embedding=embedding, 
#     persist_directory=persist_directory)  
# vectordb.persist()
# vectordb = None

# 불러오기
vectordb = Chroma(
    embedding_function=embedding, 
    persist_directory=persist_directory)  
retriever = vectordb.as_retriever()
# retriever = vectordb.as_retriever(search_kwargs={"k": 3})

qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    chain_type='stuff',
    retriever=retriever,
    # return_source_document = True
)

docs = retriever.get_relevant_documents("변동대가 설명해줘")
st.write(docs)
for doc in docs:
    # st.write(doc.metadata["source"])
    st.write(doc)

def process_llm_response(llm_response) :
    st.write('result')
    st.write(llm_response['result'])
    st.write('\n\nSources:')
    # for source in llm_response["source_documents"]:
    #     st.write(source.metadata['source'])

query = '변동대가 설명해줘'
llm_response = qa_chain(query)
process_llm_response(llm_response)
st.write(llm_response)

# query = '최 대표는 신 씨와 몇 년 일했나요?'
# llm_response = qa_chain(query)
# process_llm_response(llm_response)
# st.write(llm_response)


# model = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
# chain = RetrievalQA.from_llm(model, vectordb, return_source_documents=True)
# query = "산업은행의 중도 퇴사자 수는?"
# result = chain({"question": query, "chat_history": []})
# st.write(result)



