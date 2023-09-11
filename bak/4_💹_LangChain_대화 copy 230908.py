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
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

index_name = 'dwlangchain'
# os.environ['OPENAI_API_KEY'] = st.secrets["api_dw"]
openai.api_key = os.getenv('OPENAI_API_KEY')


def load_eco_data(question):
    query = question
    

query = "모집인원"

# 벡터DB에서 유사한 문장 가져오기 방법 1
# res = openai.Embedding.create(
#     input=[query],
#     engine='text-embedding-ada-002'
# )
# xq = res['data'][0]['embedding']
# pinecone.init(api_key=f"{st.secrets['api_pine']}", environment='gcp-starter')
# # st.write(pinecone.whoami())
# # st.write(pinecone.list_indexes())
# index = pinecone.Index(index_name)
# res = index.query(xq, top_k=2, include_metadata=True)
# # get list of retrieved text
# contexts = [item['metadata']['text'] for item in res['matches']]
# augmented_query = "\n\n---\n\n".join(contexts)+"\n\n-----\n\n"+query
# st.write(augmented_query)
# meta = [{item['metadata']['source'], item['score'], item['metadata']['text']} for item in res['matches']]
# st.write(meta)

# 벡터DB에서 유사한 문장 가져오기 방법 2
embed = OpenAIEmbeddings(
    # model='text-embedding-ada-002',
    # openai_api_key=st.secrets["api_dw"]
)
index = Pinecone.from_existing_index(index_name, embed)
similar_docs = index.similarity_search_with_score(
# similar_docs = testingIndex.similarity_search(
    query,  # our search query
    k=2
)
# st.write('similar_docs')
# st.write(similar_docs)

augmented_query = ''
for similar_doc in similar_docs:
#   st.write(similar_doc[0].page_content)
#   st.write(similar_doc[0].metadata['source'])
#   st.write(f'유사도 {similar_doc[1]}')
  augmented_query += similar_doc[0].page_content + '\n'

primer = f"""You are Q&A bot. A highly intelligent system that answers
user questions based on the information provided by the user above
each question. If the information can not be found in the information
provided by the user you truthfully say "I don't know".
"""

res = openai.ChatCompletion.create(
    # model="gpt-4",
    model='gpt-3.5-turbo',
    messages=[
        {"role": "system", "content": primer},
        {"role": "user", "content": augmented_query}
    ]
)
st.write(res['choices'][0]['message']['content'])

if "lang_messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    if message["role"] != "system": #시스템은 가리기
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.lang_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for response in openai.ChatCompletion.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        ):
            full_response += response.choices[0].delta.get("content", "")
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})


with st.expander("프롬프트 보기"):
    st.write(st.session_state)