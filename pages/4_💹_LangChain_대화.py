# https://colab.research.google.com/github/pinecone-io/examples/blob/master/docs/gpt-4-langchain-docs.ipynb#scrollTo=IThBqBi8V70d
# pip install langchain=0.0.142
# pip install openai=0.27.4
# pip install tiktoken=0.3.3
# pip install chromadb=0.3.21
# https://anpigon.tistory.com/389
import streamlit as st
import pinecone
import os
import openai
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
# from langchain.chat_models import ChatOpenAI
# from langchain.chains import RetrievalQA
# from langchain.chains import RetrievalQAWithSourcesChain
# from langchain.chains.conversation.memory import ConversationBufferWindowMemory
import convert
import numpy as np
import pandas as pd

st.title("AI DW")

if convert.check_password() == False:
    st.stop()

clear_messages_button = st.sidebar.button("Clear Conversation", key="clear")
if clear_messages_button:
    del st.session_state["lang_messages"]  # don't store password

os.environ['OPENAI_API_KEY'] = st.secrets["api_dw"]
openai.api_key = os.getenv('OPENAI_API_KEY')
pinecone.init(api_key=f"{st.secrets['api_pine']}", environment='gcp-starter')
index_name = 'dwlangchain'

def all_ids_data():
    indexquery = pinecone.Index(index_name)
    namespace = ''
    num_vectors = indexquery.describe_index_stats()["namespaces"][namespace]['vector_count']
    num_dimensions = 1536
    all_ids_df = pd.DataFrame()
    while len(all_ids_df) < num_vectors:
        input_vector = np.random.rand(num_dimensions).tolist()
        results = indexquery.query(
            vector=input_vector, 
            top_k=10000,
            include_values=False,
            include_metadata=True
            )
        for result in results['matches']:
            ids_df = pd.DataFrame([[result['id'], result['metadata']['text'], result['metadata']['source']]])
            all_ids_df = pd.concat([all_ids_df, ids_df])
    all_ids_df.reset_index(drop=True, inplace=True)
    all_ids_df.columns = ['id', 'text', 'source']
    return all_ids_df

# 벡터 중복값 제거
delete_dup_ids_button = st.sidebar.button("Delete Duplicated Ids", key="deleteadup")
if delete_dup_ids_button:
    all_ids_df = all_ids_data().sort_values(by=['text'], ascending=True)
    dup = all_ids_df.duplicated(['text'], keep='first')
    all_dup_df = pd.concat([all_ids_df, dup], axis=1)
    all_dup_df.rename(columns = {0 : 'dup'}, inplace = True)
    st.write(all_dup_df)
    remain_first_df = all_dup_df[all_dup_df['dup'] == True]['id'].values.tolist()
    st.write(remain_first_df)
    indexquery = pinecone.Index(index_name)
    indexquery.delete(ids=remain_first_df, namespace='')
    st.info(f""" 중복값을 제거하였습니다. """)  
# 벡터 전체값 제거
# delete_all_ids_button = st.sidebar.button("Delete All Ids", key="deleteall")
# if delete_all_ids_button:
#     all_ids_df = all_ids_data()
#     all_ids = all_ids_df['id'].values.tolist()
#     indexquery = pinecone.Index(index_name)
    # indexquery.delete(ids=all_ids, namespace='')
    # st.info(f""" 전체 데이터를 제거하였습니다. """)


def query_search_data(query):
    embed = OpenAIEmbeddings()
    # index = Pinecone.from_existing_index(index_name, embed)
    # 벡터DB에서 유사한 문장 가져오기 방법 1
    res = openai.Embedding.create(
        input=[query],
        engine='text-embedding-ada-002'
    )
    xq = res['data'][0]['embedding']
    indexquery = pinecone.Index(index_name)
    res = indexquery.query(xq, top_k=2, include_metadata=True)
    # st.markdown(res['matches'][0]['id'])
    query_df = pd.DataFrame()
    augmented_query = '' # 벡터DB 유사도
    for re in res['matches']:
        augmented_query += re['metadata']['text'] + '\n'
        re_df = pd.DataFrame([[re['score'], re['metadata']['text'], re['metadata']['source'], re['id']]])
        query_df = pd.concat([query_df, re_df])
        # st.markdown(re)
    query_df.reset_index(drop=True, inplace=True)
    query_df.columns = ['score', 'text', 'source', 'id']
    return augmented_query, query_df

def similarity_search_data(query):
    embed = OpenAIEmbeddings()
    index = Pinecone.from_existing_index(index_name, embed)
    similar_docs = index.similarity_search_with_score(
    # similar_docs = testingIndex.similarity_search(
        query,  # our search query
        k=2
    )
    # st.write('similar_docs')
    # st.write(similar_docs)
    # st.write(similar_docs[0][1])
    augmented_query = '' # 벡터DB 유사도    
    for similar_doc in similar_docs:
        # st.write(similar_doc[0].page_content)
        # st.write(similar_doc[0].metadata['source'])
        # st.write(f'유사도 {similar_doc[1]}')
        augmented_query += similar_doc[0].page_content + '\n'
    return augmented_query, similar_docs

if "lang_messages" not in st.session_state:
    st.session_state.lang_messages = []
    primer = f"""You are Q&A bot. A highly intelligent system that answers
    user questions based on the information provided by the user above
    each question. If the information can not be found in the information
    provided by the user you truthfully say "I don't know".
    """
    st.session_state.lang_messages.append({"role": "system", "content": primer})

for message in st.session_state.lang_messages:
    if message["role"] != "system": #시스템은 가리기
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.lang_messages.append({"role": "user", "content": prompt})
    # similarity_data, similar_datas = similarity_search_data(prompt) # 질문 채우기
    # 유사도 {similar_datas[0][1]}
    similarity_data, similar_datas = query_search_data(prompt)
    score = similar_datas['score'][0] * 100
    score = f' `유사도 {round(score, 2)}%`'
    st.session_state.lang_messages.append({"role": "user", "content": similarity_data})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for response in openai.ChatCompletion.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.lang_messages
            ],
            stream=True,
        ):
            full_response += response.choices[0].delta.get("content", "")
            message_placeholder.markdown(full_response + "▌")
        full_response += f'{score}'
        message_placeholder.markdown(full_response)
        st.write(similar_datas)
    # st.session_state.lang_messages.append({"role": "assistant", "content": full_response})

    # st.write(similar_docs[0][1])
    
    # for similar_data in similar_datas:
    #     full_response += f'/n/n 유사도 {similar_data[1]}'
        # st.session_state.lang_messages.append({"role": "assistant", "content": f'유사도 {similar_data[1]}'})
        # st.write(similar_data[0].page_content)
        # st.write(similar_data[0].metadata['source'])
        # st.write(f'유사도 {similar_data[1]}')
    # st.write(similar_datas[0].metadata['source'])

    st.session_state.lang_messages.append({"role": "assistant", "content": full_response})

with st.expander("프롬프트 보기"):
    st.write(st.session_state)
    all_ids_df = all_ids_data()
    st.write(all_ids_df)
