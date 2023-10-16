import streamlit as st
from components import apigpt
from components import convert
import openai
import json
import altair as alt
import pandas as pd

title = 'ChatGPT With Function'
# title = ''
st.set_page_config(page_title=title, page_icon="💬", layout='centered')
st.title(title)

# with st.expander("😍 무엇을 할 수 있나요?"):
info_help = '무엇을 할 수 있나요?\n\n'
info_help += '(1) 함수 호출\n\n'
info_help += '* 최신 뉴스\n\n'
info_help += '* 최근 3개월 경제지표 브리핑 해줘\n\n'
info_help += '* 최근 3개월 동종사 주가 비교 해줘\n\n'
info_help += '* 최근 3개월 주요 환율 비교 해줘\n\n'
info_help += '* 서울 날씨 어때\n\n'

info_help += '(2) 내부문서 조회\n\n'
info_help += '* 채용 규모 알고 싶어\n\n'
info_help += '* 9월 14일 이슈사항 정리해줘\n\n'
info_help += '* How to set design pressure\n\n'
info_help += '* 정보보호 주관부서 알려줘\n\n'
info_help += '* SAP ID 발급 절차 알고 싶어\n\n'
st.info(info_help, icon="😍")

if convert.check_password() == False:
    st.stop()

# 사내문서 연동여부
agree = st.sidebar.checkbox('사내문서 연동')

# 대화 초기화 
clear_button = st.sidebar.button("Clear Conversation", type="primary", key="clear")
if clear_button:
    del st.session_state.messages

if "messages" not in st.session_state:
    st.session_state.messages = []

# 대화 이력 보기
for message in st.session_state.messages:
    # if message["role"] != "system": #시스템은 가리기
    if message["role"] == "system": 
        continue
    if message["role"] == "function": 
        continue
    if message["content"] == None: 
        continue
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 질문 입력
if prompt := st.chat_input("What is up?"):

# 사용자 입력 메시지 
# 예) What's the weather like in Boston?
# 예) 오늘 날씨 어때?    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # messages = st.session_state.messages.copy()
    messages = []
    messages.append({"role": "user", "content": prompt})

    if agree: # 사내문서 연동
    # 벡터DB에서 유사 문서 가져오기
        similarity_data = []
        similar_datas = pd.DataFrame()
        similarity_data, similar_datas = apigpt.get_vector_chroma(prompt)

        with st.chat_message("assistant"):
            # st.write(similarity_data) # 유사문서 내용
            if len(similarity_data) != 0 :
                st.write('사내 유사문서가 존재합니다.') # 유사문서 리스트
                st.write(similar_datas) # 유사문서 리스트
                function_call='none'
                messages.append({"role": "user", "content": similarity_data})
            else:        
                st.write('유사 답변을 찾을수 없습니다.')
                function_call='auto'
    else: 
        function_call='auto'

    functions = apigpt.get_functions_list()

    with st.chat_message("assistant"):
        message_chart = st.empty()
        message_placeholder = st.empty()
        full_response = "" # 함수 호출이 없을 경우
        func_response = "" # 함수 호출이 있는 경우
        full_funcname = "" # 펑션 이름
        full_funcargu = "" # 펑션 파라메터
        print('==========messages==============')
        print(messages)
        print('==========st.session_state.messages==============')
        print(st.session_state.messages)
        for response in openai.ChatCompletion.create(
            # model=st.session_state["openai_model"],
            model="gpt-3.5-turbo-0613",
            # messages=[
            #     {"role": m["role"], "content": m["content"]}
            #     for m in st.session_state.messages
            # ],
            messages=messages,
            functions=functions,
            function_call=function_call,
            stream=True,
        ):
# 함수 호출이 있는 경우 (스트림)
            if response.choices[0].delta.get("function_call"):
                delta = response.choices[0].get("delta")
                function_call = response.choices[0].delta.get("function_call")
                if function_call.get("name"): 
                    func_response = delta
                else:
                    func_response.function_call.arguments += function_call.get("arguments")
                full_funcname += function_call.get("name", "")
                full_funcargu += function_call.get("arguments", "")
# 함수 호출이 없을 경우 (스트림)
            else:
                full_response += response.choices[0].delta.get("content", "")
                message_placeholder.markdown(full_response + "▌")
# 함수 호출이 없을 경우 (종료)
        if full_response:
            message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
# 함수 호출 시 (종료)
        elif func_response:
            func_messages = [{"role": "user", "content": prompt}]

            available_functions = {
                "get_economic_indicators": apigpt.get_economic_indicators,
                "get_current_weather": apigpt.get_current_weather,
                # "get_news_google": apigpt.get_news_google,
                "get_news_newsapi": apigpt.get_news_newsapi,
                "get_company_info": apigpt.get_company_info,
            }
            fuction_to_call = available_functions[full_funcname]
            function_args = json.loads(full_funcargu)
            st.write(f'{full_funcname} / {function_args}')
            # st.stop()
            if full_funcname == 'get_current_weather':
                function_response = fuction_to_call(
                    location=function_args.get("location"),
                    unit=function_args.get("unit"),
                )
                # st.session_state.messages.append({"role": "function", "name": full_funcname, "content": function_response})
                func_messages.append({"role": "function", "name": full_funcname, "content": function_response})

            elif full_funcname == 'get_economic_indicators':
                dt_range = 90
                if function_args.get("num_days"):
                    dt_range = function_args.get("num_days")
                with st.spinner(full_funcname):
                    change_eco_df, last_df = fuction_to_call(
                        num_days=int(dt_range),
                    )
                full_response = f'###### 🤖 AI 경제지표 요약 브리핑입니다. (최근 {dt_range}일)\n\n'
                message_placeholder.markdown(full_response)

                base = alt.Chart(change_eco_df).encode(x='Date:T')
                columns = sorted(change_eco_df.symbol.unique())
                selection = alt.selection_point(
                    fields=['Date'], nearest=True, on='mouseover', empty=False, clear='mouseout'
                )
                lines = base.mark_line().encode(
                    x = alt.X('Date:T', title=''),
                    y = alt.Y('rate:Q', title=''),
                    color = alt.Color('symbol:N', title='지표', legend=alt.Legend(
                        orient='bottom', 
                        direction='horizontal',
                        titleAnchor='end'))
                )
                points = lines.mark_point().transform_filter(selection)

                rule = base.transform_pivot(
                    'symbol', value='Close', groupby=['Date']
                    ).mark_rule().encode(
                    opacity=alt.condition(selection, alt.value(0.3), alt.value(0)),
                    tooltip=[alt.Tooltip(c, type='quantitative') for c in columns]
                ).add_params(selection)

                text_data = last_df
                text_data.reset_index(drop=True, inplace=True)
                text_sort_eco = text_data.sort_values(by=['rate'], ascending=False)
                text_sort_eco.reset_index(drop=True, inplace=True)
                text_data3 = pd.DataFrame(text_sort_eco.loc[0]).T
                if len(text_sort_eco.index) > 1:
                    text_data3.loc[1] = text_sort_eco.loc[len(text_sort_eco.index)-1]
                if len(text_sort_eco.index) > 2:
                    text_data3.loc[2] = text_sort_eco.loc[round(len(text_sort_eco.index)/2)]

                labels = alt.Chart(text_data3).mark_text(
                    fontWeight=600,
                    fontSize=15,
                    align='left',
                    dx=15,
                    dy=-8
                ).encode(
                    x = alt.X('Date:T', title=''),
                    y = alt.Y('rate:Q', title=''),
                    text=alt.Text('rate:Q', format='.1f'),
                    color = alt.Color('symbol:N', title='')
                )

                labels2 = alt.Chart(text_data3).mark_text(
                    fontWeight=600,
                    fontSize=15,
                    align='left',
                    dx=15,
                    dy=10
                ).encode(
                    x = alt.X('Date:T', title=''),
                    y = alt.Y('rate:Q', title=''),
                    text=alt.Text('symbol:N', title=''),
                    color = alt.Color('symbol:N', title='')
                )
                message_chart.altair_chart(lines + rule + points + labels + labels2, use_container_width=True)
                # st.altair_chart(lines + rule + points + labels + labels2, use_container_width=True)
                
                userq = '거시경제 지표 \n'
                userq += f'지표 현재가 {dt_range}일변동률''\n'
                text_sort_eco.columns = ['지표', '일자', '현재가', f'{dt_range}일변동률']
                text_sort_eco.index = text_sort_eco['지표']
                text_sort_eco.drop(['지표'], axis=1, inplace=True)
                for index, row in text_sort_eco.iterrows():
                    Close = str(round(row['현재가']))
                    rate = str(round(row[f'{dt_range}일변동률'], 2))
                    # userq = userq + '|' + index + '|' + Close + "|" + rate + '|' + '\n'
                    userq = userq + ' ' + index + ' ' + Close + " " + rate + ' ' + '\n'
                # chatGPT_msg = [{'role': 'system', 'content': '넌 대우건설 재무 분석 전문가야 경영진들에게 대우건설 주가 흐름과 거기 경제에 대해 브리핑 해줘'}]

                userq += '\n 거시경제 지표 요약하고 변동성이 큰 지표들을 과거 사례와 비교하여 경제에 미치는 영향 요약해줘'
                func_messages.append({"role": "user", "content": userq})
                # st.session_state.messages.append({"role": "assistant", "content": function_response})
            elif full_funcname == 'get_news_newsapi':
                with st.spinner(f'{full_funcname} / {function_args}'):
                    function_response = fuction_to_call(
                        search=function_args.get("search"),
                        numOfRows=function_args.get("numOfRows")
                    )
                cols = st.columns(5)
                if function_args.get("search") :
                    for idx, article in enumerate(function_response): 
                        colsnum = ( idx % 5 )
                        with cols[colsnum]:
                            st.link_button(f"{article['title']}", f"{article['link']}")
                else :
                    
                    for idx, article in enumerate(function_response): 
                        colsnum = ( idx % 5 )
                        with cols[colsnum]:
                            if article['urlToImage']:
                                urlToImage = f'<a href = "{article["url"]}" style="text-decoration:none"><img src="{article["urlToImage"]}" width="100%"><br>{article["title"]}</a>'
                                st.write(urlToImage, unsafe_allow_html=True)
                            else:
                                urlToImage = f'<a href = "{article["url"]}" style="text-decoration:none"><img src="https://newsapi.org/images/flags/kr.svg" width="100%"><br>{article["title"]}</a>'
                                st.write(urlToImage, unsafe_allow_html=True)

            elif full_funcname == 'get_news_google':
                with st.spinner(f'{full_funcname} / {function_args}'):
                    function_response = fuction_to_call(
                        country=function_args.get("country"),
                        numOfRows=function_args.get("numOfRows")
                    )

                f'{full_funcname} / {function_args}'
                for entry in function_response.entries[:100]:
                    st.link_button(f"{entry.title}", f"{entry.link}")
                    print(entry.title)
                    # st.write(f'{entry.published} {entry.title}')
                    # 
                    # "제목:", entry.title
                    # "일자:", 
                    # "링크:", entry.link
                # st.markdown(function_response)
                
                # func_messages.append({"role": "function", "name": full_funcname, "content": function_response})
                # full_response = f'###### 🤖 AI 경제지표 요약 브리핑입니다. (최근 {dt_range}일)\n\n'
                # message_placeholder.markdown(full_response)
            elif full_funcname == 'get_company_info':
                function_response = fuction_to_call(
                    company=function_args.get("company")
                )
                func_messages.append({"role": "function", "name": full_funcname, "content": function_response})

            # print(func_messages)
            st.stop()
            for response in openai.ChatCompletion.create(
                # model=st.session_state["openai_model"],
                model="gpt-3.5-turbo-0613",
                messages=func_messages,
                stream=True,
            ):
                full_response += response.choices[0].delta.get("content", "")
                message_placeholder.markdown(full_response + "▌")

            full_response += f"""
\n\n
```
* 함수: {full_funcname}
* 변수: {full_funcargu}
* 응답: {func_response}
```
\n\n
"""

            message_placeholder.markdown(full_response)

            st.session_state.messages.append({"role": "assistant", "content": full_response})

with st.expander("프롬프트 보기"):
    st.write(st.session_state.messages)

clear_button = st.button("대화 초기화", type="primary", key="clear2")
if clear_button:
    # del st.session_state.messages
    st.session_state.messages = []
    st.rerun()

