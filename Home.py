import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
import altair as alt
import openai
import convert

st.header("일하기 좋은 회사 1위 대우건설 VS 동종사 👋 ")

progress_stock = st.progress(0) # 주가정보 로딩바
status_stock = st.empty() # 주가정보 로딩바

st.write(""" ### 🤖 AI 브리핑 """)
ai_eco_text = st.empty()
ai_stock_text = st.empty()

dt_range = st.sidebar.radio('기간', ['3개월', '6개월', '1년', '3년', '10년'])
if dt_range == '1개월':
    start_date = st.sidebar.date_input('Start date', datetime.today() - relativedelta(months=1))
elif dt_range == '3개월':
    start_date = st.sidebar.date_input('Start date', datetime.today() - relativedelta(months=3))
elif dt_range == '6개월':    
    start_date = st.sidebar.date_input('Start date', datetime.today() - relativedelta(months=6))
elif dt_range == '1년':    
    start_date = st.sidebar.date_input('Start date', datetime.today() - relativedelta(years=1))
elif dt_range == '3년':    
    start_date = st.sidebar.date_input('Start date', datetime.today() - relativedelta(years=3))
elif dt_range == '10년':    
    start_date = st.sidebar.date_input('Start date', datetime.today() - relativedelta(years=10))
end_date = datetime.today()

### 사이드바 종목 설정 #########################################################
stocks = [
    {'name': ' 대우건설', 'symbol': '047040.KS'}
    ]

multi_stocks = st.sidebar.multiselect(
    "동종사를 선택하세요",
    [
        # "인선이엔티 060150.KQ",
        # "코웨이 021240.KS",
        "삼성물산 028260.KS",
        "현대건설 000720.KS",
        "DL이앤씨 375500.KS",
        "GS건설 006360.KS",
        "삼성엔지니어링 028050.KS",
        "HDC현대산업개발 294870.KS",
        "금호건설 002990.KS"
        ],
    [ #초기 선택
        # "인선이엔티 060150.KQ",
        # "코웨이 021240.KS",
        # "삼성물산 028260.KS",
        "HDC현대산업개발 294870.KS",
        "GS건설 006360.KS",
        "현대건설 000720.KS",
        "DL이앤씨 375500.KS"
        ]
    )

for stock in multi_stocks:
    words = stock.split()
    stocks.append({'name': words[0], 'symbol': words[1]})

change_stocks_df = pd.DataFrame() # 주가 변동률
info_stock_df = pd.DataFrame() # 주가 변동률
# info_stock_df = pd.DataFrame(
#     index=['시가총액', 
#            '시가총액(억)', 
#            '매수의견', 
#            '현재가', 
#            '총현금액',
#            '총부채액',
#            '총매출액',
#            '매출총이익', 
#            '영업이익률',
#         #    '순이익률',
#             '변동률',
#            '비고']
# )

##########################################################################
### 주가정보 불러오기 ########################################################
##########################################################################
for i, stock in enumerate(stocks):
    l_rate = round(i / len(stocks) * 100)
    progress_stock.progress(l_rate)
    status_stock.text("1/2 주가정보를 불러오는 중입니다... %i%%" % l_rate)

    get_stock_data = yf.Ticker(stock['symbol'])
    stock_df = get_stock_data.history(period='1d', start=start_date, end=end_date)
    # 일간변동률, 누적합계
    stock_df['dpc'] = (stock_df.Close/stock_df.Close.shift(1)-1)*100
    stock_df['cs'] = stock_df.dpc.cumsum()
    
    change2_df = pd.DataFrame(
        {
            'symbol': stock['name'],
            'rate': stock_df.cs,
            }
    )

    change2_df.reset_index(drop=False, inplace=True)
    change_stocks_df = pd.concat([change_stocks_df, change2_df])

# prompt = respense["choices"][0].get("delta", {}).get("content")
    info_stock_df[stock['name']] = [
        get_stock_data.info['marketCap'],
        convert.get_kor_amount_string_no_change(get_stock_data.info['marketCap']),
        get_stock_data.info['recommendationKey'],
        get_stock_data.info['currentPrice'],
        get_stock_data.info['totalCash'], # 총현금액
        get_stock_data.info['totalDebt'], # 총부채액
        get_stock_data.info['totalRevenue'], # 총매출액
        get_stock_data.info.get('grossProfits', ''), # 매출총이익
        get_stock_data.info['operatingMargins'] * 100, # 영업이익률
        round(change_stocks_df[-1:].iloc[0]['rate'], 1), # 변동률
        '']

rate_text = f'{dt_range}변동률'
info_stock_df.index = [
    '시가총액', 
    '시가총액(억)', 
    '매수의견', 
    '현재가', 
    '총현금액',
    '총부채액',
    '총매출액',
    '매출총이익', 
    '영업이익률',
#    '순이익률',
    rate_text,
    '비고'
    ]

##########################################################################
### 주가정보 차트그리기 #######################################################
##########################################################################
st.write(f""" ### 🚀 {dt_range} 누적변동률  """)

line_chart = alt.Chart(change_stocks_df).mark_line().encode(
    x = alt.X('Date:T', title=''),
    y = alt.Y('rate:Q', title=''),
    color = alt.Color('symbol:N', title='', legend=alt.Legend(
        orient='bottom',
        direction='horizontal',
        titleAnchor='end'))
)

text_data = change_stocks_df.loc[change_stocks_df['Date'].idxmax()]
text_data.reset_index(drop=True, inplace=True)
text_data2 = text_data.sort_values(by=['rate'], ascending=True)
text_data2.reset_index(drop=True, inplace=True)
text_data3 = pd.DataFrame(text_data.loc[0]).T
if len(text_data2.index) > 1:
    text_data3.loc[1] = text_data2.loc[0]
if len(text_data2.index) > 2:
    text_data3.loc[2] = text_data2.loc[round(len(text_data3.index)/2)]

# rate_text = f'{dt_range}변동률'
labels = alt.Chart(text_data3).mark_text(
    # point=True,
    fontWeight=600,
    fontSize=13,
    # color='white',
    align='left',
    dx=15,
    dy=-10
).encode(
    x = alt.X('Date:T', title=''),
    # y = alt.Y('rate:Q', title='변동률'),
    y = alt.Y('rate:Q', title=rate_text),
    # y = 'rate:Q',
    text=alt.Text('rate:Q', format='.1f'),
    color = alt.Color('symbol:N', title='')
)

labels2 = alt.Chart(text_data3).mark_text(
    # point=True,
    fontWeight=600,
    fontSize=14,
    # color='white',
    align='left',
    dx=15,
    dy=8
).encode(
    x = alt.X('Date:T', title=''),
    y = alt.Y('rate:Q', title=rate_text),
    # y = 'rate:Q',
    text=alt.Text('symbol:N'),
    color = alt.Color('symbol:N', title='')
)
st.altair_chart(line_chart + labels + labels2, use_container_width=True)

st.write(""" ### 🎙️ 시가총액 """)
# cap_df = info_stock_df.T
cap_df = info_stock_df.iloc[[0, 1]].T #시가총액, 시가총액(억)
cap_df.reset_index(drop=False, inplace=True)
cap_df.rename(columns={'index': '종목명'}, inplace=True)
bar_chart = alt.Chart(cap_df, title='').mark_bar().encode(
                x = alt.X('시가총액:Q', title='', axis=alt.Axis(labels=False)),
                y = alt.Y('종목명:O', title=''),
                color = alt.Color('종목명:N', title='종목', legend=None)   
            )

bar_text = alt.Chart(cap_df).mark_text(
    fontWeight=600,
    fontSize=14,
    align='left',
    dx=10,
    dy=1
    ).transform_calculate(
    text_mid = '(datum.b/2)').encode(
                x=alt.X('시가총액:Q', title='', axis=alt.Axis(labels=False)),
                y=alt.Y('종목명:O'),
                # detail='TERMS:N',
                # text=alt.Text('시가총액:Q', format='.0f')
                color = alt.Color('종목명:N', title=''),
                text=alt.Text('시가총액(억):N')
            )
st.altair_chart(bar_chart + bar_text, use_container_width=True)

### 사이드바 종목 설정 #########################################################
products = [
    {'name': ' 원/달러', 'symbol': 'USDKRW=X'}
    ]

multi_products = st.sidebar.multiselect(
    "지표를 선택하세요",
    [
        "크루드오일 CL=F",
        "금 GC=F",
        "은 SI=F",
        # "구리 GH=F",
        "S&P500 ^GSPC",
        "천연가스 LNG",
        "10년물 ^TNX",
        "DBC DBC",
        "BTC-USD BTC-USD",
        "달러인덱스 DX-Y.NYB"
        ],
    [ #초기 선택
        "크루드오일 CL=F",
        "금 GC=F",
        "은 SI=F",
        # "구리 GH=F",
        "S&P500 ^GSPC",
        "천연가스 LNG",
        "10년물 ^TNX",
        "DBC DBC",
        "BTC-USD BTC-USD",
        "달러인덱스 DX-Y.NYB"
        ]
    )

##########################################################################
### 경제지표 블러오기 #########################################################
##########################################################################

for product in multi_products:
    words = product.split()
    products.append({'name': words[0], 'symbol': words[1]})

change_eco_df = pd.DataFrame() # 변동률
last_df = pd.DataFrame() # 변동률

# with st.spinner(text="각종 지표 불러오는중..."):    
for idx, product in enumerate(products):

    l_rate = round(i / len(products) * 100)
    progress_stock.progress(l_rate)
    status_stock.text("2/2 지표정보를 불러오는 중입니다... %i%%" % l_rate)

    get_product_data = yf.Ticker(product['symbol'])
    product_df = get_product_data.history(period='1d', start=start_date, end=end_date)

    # 일간변동률, 누적합계
    product_df['dpc'] = (product_df.Close/product_df.Close.shift(1)-1)*100
    product_df['cs'] = product_df.dpc.cumsum()

    change2_df = pd.DataFrame(
        {
            'symbol': product['name'],
            'Close': product_df.Close,
            'rate': product_df.cs,
            }
    )
    change2_df.reset_index(drop=False, inplace=True)
    change_eco_df = pd.concat([change_eco_df, change2_df])

    last2_df = pd.DataFrame(product_df.iloc[len(product_df.index)-1]).T
    last3_df = pd.DataFrame(
        {
            'symbol': product['name'],
            'Date': last2_df.index,
            'Close': last2_df.Close, 
            'rate': last2_df.cs,
            }
    )
    last_df = pd.concat([last_df, last3_df])

##########################################################################
### 경제지표 차트그리기 #######################################################
##########################################################################
status_stock.text("")
progress_stock.empty()
st.write(f""" ### 📈 {dt_range} 지표변동률  """)

line_chart = alt.Chart(change_eco_df).mark_line().encode(
    x = alt.X('Date:T', title=''),
    y = alt.Y('rate:Q', title=''),
    # color = alt.Color('symbol:N', title='종목', legend=None)
    color = alt.Color('symbol:N', title='', legend=alt.Legend(
        orient='bottom', #none
        # legendX=130, legendY=0,
        direction='horizontal',
        titleAnchor='end'))
)

text_data = last_df
text_data.reset_index(drop=True, inplace=True)
text_data2 = text_data.sort_values(by=['rate'], ascending=True)
text_data2.reset_index(drop=True, inplace=True)
text_data3 = pd.DataFrame(text_data2.loc[0]).T
if len(text_data2.index) > 1:
    text_data3.loc[1] = text_data2.loc[len(text_data2.index)-1]
if len(text_data2.index) > 2:
    text_data3.loc[2] = text_data2.loc[round(len(text_data2.index)/2)]

labels = alt.Chart(text_data3).mark_text(
    # point=True,
    fontWeight=600,
    fontSize=14,
    # color='white',
    align='left',
    dx=15,
    dy=-8
).encode(
    x = alt.X('Date:T', title=''),
    y = alt.Y('rate:Q', title=rate_text),
    # y = 'rate:Q',
    text=alt.Text('rate:Q', format='.1f'),
    color = alt.Color('symbol:N', title='')
)

labels2 = alt.Chart(text_data3).mark_text(
    # point=True,
    fontWeight=600,
    fontSize=13,
    # color='white',
    align='left',
    dx=15,
    dy=10
).encode(
    x = alt.X('Date:T', title=''),
    y = alt.Y('rate:Q', title=rate_text),
    text=alt.Text('symbol:N'),
    color = alt.Color('symbol:N', title='')
)

st.altair_chart(line_chart + labels + labels2, use_container_width=True)

##########################################################################
##########################################################################
##########################################################################
openai.api_key = st.secrets["api_key"]

##########################################################################
### AI 경제지표 브리핑 #######################################################
##########################################################################
eco_msg = [{'role': 'system', 'content': '넌 재무 분석가야'}]

userq = f'|지표|현재가|{dt_range}변동률|' + '\n'

# DataFrame의 각 행을 ChatCompletion messages에 추가
for index, row in last_df.iterrows():
    Close = str(round(row['Close']))
    rate = str(round(row['rate'], 2))
    userq = userq + '|' + row['symbol'] + '|' + Close + "|" + rate + '|' + '\n'
userq += '요약은 하지말고 현재 경제상황을 전문적으로 설명하고 과거 유사한 사례가 있으면 알려주고 앞으로의 경제상황 예측해줘'
user_message = {'role': 'user', 'content': f"{userq}"}
eco_msg.extend([user_message])

streamText = '🤖 '
# with st.spinner('1) Waiting for ChatGPT...'):
print(eco_msg)
get_respense = openai.ChatCompletion.create(
    model = "gpt-3.5-turbo",
    messages = eco_msg,
    temperature=0,
    stream=True,   
)

for respense in get_respense:
    # prompt = respense["choices"][0]["message"]["content"]
    prompt = respense["choices"][0].get("delta", {}).get("content")
    if prompt is not None:
        streamText = streamText + prompt
        ai_eco_text.info(f""" {streamText} """)
        # print(prompt, end='') # 한줄씩 츨략
        # print(prompt, end='') # 한줄씩 츨략

##########################################################################
### AI 동종사 비교 ##########################################################
##########################################################################
# DataFrame 결과를 ChatCompletion messages에 넣기 위한 변환
# messages = [{'role': 'system', 'content': '넌 대우건설 재무 분석가야'},
#             {'role': 'assistant', 'content': '비교 분석해줘'}]
stock_msg = [{'role': 'system', 'content': '넌 대우건설 재무 분석가야'}]

# st.write(info_stock_df)
chat_df = info_stock_df.T
# st.write(chat_df)
userq = '|회사명|현재가|매수의견|시가총액||변동률| \n'
# DataFrame의 각 행을 ChatCompletion messages에 추가
for index, row in chat_df.iterrows():
    userq += '|' + index + '|' + str(round(row['현재가'])) + '|' + row['매수의견'] + '|' 
    userq += row['시가총액(억)'] + '|' + str(row[rate_text]) + '|' + '\n' 
userq += '50글자로 분석해줘'
user_message = {'role': 'user', 'content': f"{userq}"}
stock_msg.extend([user_message])
# user_message = {'role': 'user', 'content': "50글자로 분석해줘"}
# messages.extend([user_message])

streamText = '🤖 '
print(stock_msg)
get_respense = openai.ChatCompletion.create(
    model = "gpt-3.5-turbo",
    messages = stock_msg,
    # temperature=0,
    stream=True,
)
for respense in get_respense:
    prompt = respense["choices"][0].get("delta", {}).get("content")
    if prompt is not None:
        streamText = streamText + prompt
        ai_stock_text.success(f""" {streamText} """)       
        # print(prompt, end='') # 한줄씩 츨려ㄱ

with st.expander("프롬프트 보기"):
    st.write(cap_df) # 시가총액, 현재가
    st.write(text_data) # 변동률
    st.write(last_df)

    st.write(stock_msg)
    st.write(eco_msg)
    

