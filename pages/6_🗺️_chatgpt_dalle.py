import streamlit as st
import openai
import pandas as pd

openai.api_key = st.secrets["api_key"]

with st.form('form'):
  user_input = st.text_input('Prompt')
  size = st.selectbox("Size", ["1024x1024", "512x512", "256x256"])
  submit = st.form_submit_button('Submit')

if submit and user_input:
  gpt_prompt = [{
    "role": "system",
    "content": "Imagine the detail appeareance of the input, Response it shortly around 20 words."
  }]
  
  gpt_prompt.append({
    "role": "user", 
    "content": user_input
  })

  with st.spinner('Waiting for ChatGPT...'):
    get_respense = openai.ChatCompletion.create(
      model = "gpt-3.5-turbo",
      messages = gpt_prompt
    )

  prompt = get_respense["choices"][0]["message"]["content"]

  # st.write(user_input)
  # st.write(openai.api_key)
  st.write(prompt)

  with st.spinner('Waiting for DALL_E...'):
    dalle_response = openai.Image.create(
      prompt=prompt,
      size=size
    )

  st.image(dalle_response["data"][0]["url"])

# 예시로 사용할 DataFrame 생성
data = {'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, 22]}
df = pd.DataFrame(data)

# DataFrame 결과를 ChatCompletion messages에 넣기 위한 변환
messages = [{'role': 'system', 'content': 'You are a user'},
            {'role': 'assistant', 'content': 'Hello! I can help you with your DataFrame results.'}]

# DataFrame의 각 행을 ChatCompletion messages에 추가
for index, row in df.iterrows():
    user_message = {'role': 'user', 'content': f"Tell me about {row['Name']}'s age."}
    assistant_message = {'role': 'assistant', 'content': f"{row['Name']} is {row['Age']} years old."}
    messages.extend([user_message, assistant_message])

# 최종적으로 생성된 messages 출력
st.write(messages)

# ChatGPT API에 메시지 전송
response = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=messages
)

# 최종적으로 생성된 대화 로그 출력
# for message in response['choices'][0]['message']['content']:
    # st.write(message)

st.write(response['choices'][0]['message']['content'])
