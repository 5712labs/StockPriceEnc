import streamlit as st
import openai
import convert

st.title("ChatGPT4")

if convert.check_password() == False:
    st.stop()
 
clear_button = st.sidebar.button("Clear Conversation", key="clear")
if clear_button:
#     st.session_state['generated'] = []
#     st.session_state['past'] = []
    st.session_state['messages'] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
#     st.session_state['number_tokens'] = []
#     st.session_state['model_name'] = []
#     st.session_state['cost'] = []
#     st.session_state['total_cost'] = 0.0
#     st.session_state['total_tokens'] = []
#     counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")

if "messages" not in st.session_state:
    st.session_state.messages = []
    # st.write("messages not in st.session_state")
    # st.write(st.session_state.messages)

for message in st.session_state.messages:
    if message["role"] != "system": #시스템은 가리기
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
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