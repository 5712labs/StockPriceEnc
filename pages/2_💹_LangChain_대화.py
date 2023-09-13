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
from langchain.vectorstores import Chroma
# from langchain.vectorstores import Pinecone
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

# Create Radio Buttons
llm_db_type = st.radio(label = 'select llm db ', options = ['Pinecone', 'Chroma', 'FAISS'], index=0)
st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

os.environ['OPENAI_API_KEY'] = st.secrets["api_dw"]
openai.api_key = os.getenv('OPENAI_API_KEY')

clear_messages_button = st.sidebar.button("Clear Conversation", key="clear")
if clear_messages_button:
    del st.session_state["plang_messages"]  # don't store password
    del st.session_state["clang_messages"]  # don't store password

# 1. Prompt의 범위를 너무 넓게 잡지 않기
# 2. Prompt의 재현율이 낮으면 act, example로 원하는 행태의 결과를 재현할 수 있도록 Prompt 추가
# 3. 사용하는 chain에 대한 이해 (help() 나 langchain 공식문서)
# 4. 잘 만든 Prompt에 추가적인 기능을 넣고 싶다면 FakeAgent
# 5. Chroma는 생각보다 강력한 기능이 있습니다.
# 6. 잘 만들어진 다양한 Prompt 보기
# https://github.com/f/awesome-chatgpt-prompts

primer = f""" You are Q&A bot. A highly intelligent system that answers 
user questions based on the information provided by the user above each question.
To answer the question at the end, use the following context.
If the information can not be found in the information provided by the user you truthfully say "I don't know".
you only answer in Korean
(summaries)
"""

system_template=f""" To answer the question at the end, use the following context. 
If you don't know the answer, just say you don't know and don't try to make up an answer.
I want you to act as my Burger King menu recommender. It tells you your budget and suggests what to buy. 
You should only reply to items you recommend. Don't write a description.

Below is an example.
"My budget is 10,000 won, and it is the best menu combination within the budget."
you only answer in Korean
(summaries)
"""

if "plang_messages" not in st.session_state:
    st.session_state.plang_messages = []
    st.session_state.plang_messages.append({"role": "system", "content": primer, "from": "sys"})

if "clang_messages" not in st.session_state:
    st.session_state.clang_messages = []
    st.session_state.clang_messages.append({"role": "system", "content": primer, "from": "sys"})

if llm_db_type == 'Pinecone':
    for message in st.session_state.plang_messages:
        if message["role"] != "system": #시스템은 가리기
            if message["from"] != "db": 
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
    pinecone.init(api_key=f"{st.secrets['api_pine']}", environment='gcp-starter')
    index_name = 'dwlangchain'

    def query_search_data(query):
        embed = OpenAIEmbeddings()
        # 벡터DB에서 유사한 문장 가져오기 방법 1
        res = openai.Embedding.create(
            input=[query],
            engine='text-embedding-ada-002'
        )
        xq = res['data'][0]['embedding']
        indexquery = pinecone.Index(index_name)
        res = indexquery.query(xq, top_k=2, include_metadata=True)
        # st.markdown(res)
        # matches: [{id:''},
        #           {metadata:{source:'', text:''}},
        #           {score:''},
        #           ]
        # st.markdown(res['matches'][0]['id'])
        query_df = pd.DataFrame()
        augmented_query = '' # 벡터DB 유사도
        # st.write(res['matches'])
        for re in res['matches']:
            if re['score'] < 0.8:
                continue
            augmented_query += re['metadata']['text'] + '\n'
            re_df = pd.DataFrame([[re['score'], re['metadata']['text'], re['metadata']['source'], re['id']]])
            # st.markdown(re_df)
            query_df = pd.concat([query_df, re_df])
            # st.markdown(re)
        if len(query_df) > 0 :
            query_df.reset_index(drop=True, inplace=True)
            query_df.columns = ['score', 'text', 'source', 'id']
        # st.write(augmented_query)
        # st.write(query_pinecon_df)
        return augmented_query, query_df

    # def similarity_search_data(query):
    #     embed = OpenAIEmbeddings()
    #     index = Pinecone.from_existing_index(index_name, embed)
    #     similar_docs = index.similarity_search_with_score(
    #     # similar_docs = testingIndex.similarity_search(
    #         query,  # our search query
    #         k=2
    #     )
    #     # st.write('similar_docs')
    #     # st.write(similar_docs)
    #     # st.write(similar_docs[0][1])
    #     augmented_query = '' # 벡터DB 유사도    
    #     for similar_doc in similar_docs:
    #         # st.write(similar_doc[0].page_content)
    #         # st.write(similar_doc[0].metadata['source'])
    #         # st.write(f'유사도 {similar_doc[1]}')
    #         augmented_query += similar_doc[0].page_content + '\n'
    #     return augmented_query, similar_docs

    if prompt := st.chat_input("What is up?"):
        if len(st.session_state.plang_messages) >= 3 :
            for lang_messages in st.session_state.plang_messages:
                st.write(lang_messages)
            st.session_state.plang_messages.pop(-2)
        st.session_state.plang_messages.append({"role": "user", "content": prompt, "from": "input"})
        # similarity_data, similar_datas = similarity_search_data(prompt) # 질문 채우기
        # 유사도 {similar_datas[0][1]}
        similarity_data = []
        similar_datas = pd.DataFrame()
        similarity_data, similar_datas = query_search_data(prompt)
        # st.write(similarity_data)
        # st.write(similar_datas)
        score = ''
        if len(similar_datas) != 0:
            score = similar_datas['score'][0] * 100
            similar_datas['score'] = f'{round(score, 2)}%'
            score = f' `유사도 {round(score, 2)}%`'
            
        st.session_state.plang_messages.append({"role": "user", "content": similarity_data, "from": "db"})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if len(similar_datas) == 0 :
                st.write('유사 답변을 찾을수 없습니다.')
            else :
                similar_datas.columns = ['유사도', '본문 내용', '출처', 'id']
                st.write(similar_datas)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for response in openai.ChatCompletion.create(
                model=st.session_state["openai_model"],
                messages=[
                    {"role": message["role"], "content": message["content"]}
                    for message in st.session_state.plang_messages
                ],
                stream=True,
            ):
                full_response += response.choices[0].delta.get("content", "")
                message_placeholder.markdown(full_response + "▌")
            full_response += f'{score}'
            message_placeholder.markdown(full_response)
            # if len(similar_datas) == 0 :
            #     st.write('유사 답변을 찾을수 없습니다.')
            # else :
            #     st.write(similar_datas)
            st.session_state.plang_messages.append({"role": "assistant", "content": full_response, "from": "chatGPT"})
            full_response = ''
            score = ''
            
            # aaa = {key: value for key, value in a.items() if value != 2}  ## 값이 2인것 빼고(1,3,4) a변수에 딕셔너리를 생성


    # st.session_state.plang_messages.append({"role": "user", "content": similarity_data, "from": "db"})



        # st.session_state.plang_messages.append({"role": "assistant", "content": full_response})

        # st.write(similar_docs[0][1])
        
        # for similar_data in similar_datas:
        #     full_response += f'/n/n 유사도 {similar_data[1]}'
            # st.session_state.plang_messages.append({"role": "assistant", "content": f'유사도 {similar_data[1]}'})
            # st.write(similar_data[0].page_content)
            # st.write(similar_data[0].metadata['source'])
            # st.write(f'유사도 {similar_data[1]}')
        # st.write(similar_datas[0].metadata['source'])
        # st.write(st.session_state)
        # st.session_state.plang_messages.pop(1)

        # st.session_state.plang_messages.append({"role": "assistant", "content": full_response, "from": "chatGPT"})

        # all_ids_df = all_ids_data()
        # st.write(all_ids_df)

elif llm_db_type == 'Chroma':
    for message in st.session_state.clang_messages:
        if message["role"] != "system": #시스템은 가리기
            if message["from"] != "db": 
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
    persist_directory="db"
    embedding = OpenAIEmbeddings()
    vectordb = Chroma(
        embedding_function=embedding, 
        persist_directory=persist_directory)  
    # st.write(vectordb.get())
    # st.write(vectordb.get().keys())
    # st.write(len(vectordb.get()["ids"]))
    # retriever = vectordb.as_retriever()
    # retriever = vectordb.as_retriever(search_kwargs={"k": 2})
    vectordb.as_retriever(search_kwargs={"k": 2})
    
    def query_search_chroma(query):
        docs = vectordb.similarity_search_with_relevance_scores(query)
        # docs2 = vectordb.similarity_search_by_vector_with_relevance_scores(query)
        # st.markdown(docs2)
        # Document: [{page_content:''},
        #           {metadata:{source:'', text:''}},
        #           {score:''},
        #           ]

        query_df = pd.DataFrame()
        augmented_query = '' # 벡터DB 유사도
        for doc in docs:
            if doc[1] < 0.7:
                continue
            # augmented_query += doc['metadata']['text'] + '\n'
            augmented_query += doc[0].page_content + '\n'
            # re_df = pd.DataFrame([[doc[1], doc['metadata']['text'], doc['metadata']['source'], doc['id']]])
            re_df = pd.DataFrame([[doc[1], doc[0].page_content, doc[0].metadata['source'], 'IDIDID']])
            query_df = pd.concat([query_df, re_df])
            # st.markdown(re)
            # for doc in docs:
            #     st.markdown(doc)
                # st.markdown(doc[1]) # 유시도
                # st.markdown(doc[0].metadata["source"])
                # st.markdown(doc[0].page_content)
        if len(query_df) > 0 :
            query_df.reset_index(drop=True, inplace=True)
            query_df.columns = ['score', 'text', 'source', 'id']
        return augmented_query, query_df

    if prompt := st.chat_input("What is up?"):
        if len(st.session_state.clang_messages) >= 3 :
            st.session_state.clang_messages.pop(-2)
        st.session_state.clang_messages.append({"role": "user", "content": prompt, "from": "input"})
        similarity_data = []
        similar_datas = pd.DataFrame()
        similarity_data, similar_datas = query_search_chroma(prompt)
        # st.write(similarity_data)
        # st.write(similar_datas)
        score = ''
        if len(similar_datas) != 0:
            score = similar_datas['score'][0] * 100
            score = f' `유사도 {round(score, 2)}%`'
        st.session_state.clang_messages.append({"role": "user", "content": similarity_data, "from": "db"})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if len(similar_datas) == 0 :
                st.write('유사 답변을 찾을수 없습니다.')
            else :
                similar_datas.columns = ['유사도', '본문 내용', '출처', 'id']
                st.write(similar_datas)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for response in openai.ChatCompletion.create(
                model=st.session_state["openai_model"],
                messages=[
                    {"role": message["role"], "content": message["content"]}
                    for message in st.session_state.clang_messages
                ],
                stream=True,
            ):
                full_response += response.choices[0].delta.get("content", "")
                message_placeholder.markdown(full_response + "▌")
            full_response += f'{score}'
            message_placeholder.markdown(full_response)
            # if len(similar_datas) == 0 :
            #     st.write('유사 답변을 찾을수 없습니다.')
            # else :
            #     st.write(similar_datas)
            st.session_state.clang_messages.append({"role": "assistant", "content": full_response, "from": "chatGPT"})
            full_response = ''
            score = ''
            

    # docs = retriever.get_relevant_documents("대우건설")
    # for doc in docs:
    #     # st.write(doc.metadata["source"])
    #     st.markdown(doc)

    # docs = vectordb.similarity_search_with_score('SAP ID 발급 절차 알고 싶어')
    # docs = vectordb.similarity_search_with_score('재무회계시스템') #낮을수록 좋음
    # for doc in docs:
    # #     # st.markdown(doc)
    #     st.markdown(doc[1]) # 유시도
    #     st.write(doc[0].metadata["source"])
    #     st.markdown(doc[0].page_content)


elif llm_db_type == 'FAISS':    
    st.write('')







with st.expander("프롬프트 보기"):
    st.write(st.session_state)

clear_messages_bottom = st.button("Clear Conversation", key="clearlang", type='secondary')
if clear_messages_bottom:
    del st.session_state["plang_messages"]  # don't store password
    del st.session_state["clang_messages"]  # don't store password
