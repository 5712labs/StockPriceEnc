import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
import altair as alt
import openai

st.header("일하기 좋은 회사 1위 대우건설 VS 동종사 👋 ")

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
        "GS건설 006360.KS",
        "현대건설 000720.KS",
        "DL이앤씨 375500.KS"
        ]
    )

for stock in multi_stocks:
    words = stock.split()
    stocks.append({'name': words[0], 'symbol': words[1]})

### 공통함수 ###############################################################
def get_kor_amount_string(num_amount, ndigits_round=0, str_suffix='원'):
    """숫자를 자릿수 한글단위와 함께 리턴한다 """
    assert isinstance(num_amount, int) and isinstance(ndigits_round, int)
    assert num_amount >= 1, '최소 1원 이상 입력되어야 합니다'
    ## 일, 십, 백, 천, 만, 십, 백, 천, 억, ... 단위 리스트를 만든다.
    maj_units = ['만', '억', '조', '경', '해', '자', '양', '구', '간', '정', '재', '극'] # 10000 단위
    units     = [' '] # 시작은 일의자리로 공백으로하고 이후 십, 백, 천, 만...
    for mm in maj_units:
        units.extend(['십', '백', '천']) # 중간 십,백,천 단위
        units.append(mm)
    
    list_amount = list(str(round(num_amount, ndigits_round))) # 라운딩한 숫자를 리스트로 바꾼다
    list_amount.reverse() # 일, 십 순서로 읽기 위해 순서를 뒤집는다
    
    str_result = '' # 결과
    num_len_list_amount = len(list_amount)
    
    for i in range(num_len_list_amount):
        str_num = list_amount[i]
        # 만, 억, 조 단위에 천, 백, 십, 일이 모두 0000 일때는 생략
        if num_len_list_amount >= 9 and i >= 4 and i % 4 == 0 and ''.join(list_amount[i:i+4]) == '0000':
            continue
        if str_num == '0': # 0일 때
            if i % 4 == 0: # 4번째자리일 때(만, 억, 조...)
                str_result = units[i] + str_result # 단위만 붙인다
        elif str_num == '1': # 1일 때
            if i % 4 == 0: # 4번째자리일 때(만, 억, 조...)
                str_result = str_num + units[i] + str_result # 숫자와 단위를 붙인다
            else: # 나머지자리일 때
                str_result = units[i] + str_result # 단위만 붙인다
        else: # 2~9일 때
            str_result = str_num + units[i] + str_result # 숫자와 단위를 붙인다
    str_result = str_result.strip() # 문자열 앞뒤 공백을 제거한다 
    if len(str_result) == 0:
        return None
    if not str_result[0].isnumeric(): # 앞이 숫자가 아닌 문자인 경우
        str_result = '1' + str_result # 1을 붙인다
    return str_result + str_suffix # 접미사를 붙인다

def get_kor_amount_string_no_change(num_amount, ndigits_keep=3):
    """잔돈은 자르고 숫자를 자릿수 한글단위와 함께 리턴한다 """
    return get_kor_amount_string(num_amount, -(len(str(num_amount)) - ndigits_keep))
# st.write(get_kor_amount_string(12345))
# st.write(get_kor_amount_string_no_change(123456789))

change_df = pd.DataFrame() # 변동률
rate_df = pd.DataFrame() # 변동률

info_df = pd.DataFrame(
    index=['시가총액', 
           '시가총액변환', 
           '매수의견', 
           '현재가', 
        #    '총현금액', 
        #    '총부채액', 
        #    '총매출액',
        #    '매출총이익', 
        #    '영업이익률',
        #    '순이익률', 
           '비고']
)

progress_bar = st.progress(0)
status_text = st.empty()

for i, stock in enumerate(stocks):
    l_rate = round(i / len(stocks) * 100)
    progress_bar.progress(l_rate)
    # status_text.text("%i%% Complete" % l_rate)
    status_text.text("주가정보를 불러오는 중입니다. %i%%" % l_rate)

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
    change_df = pd.concat([change_df, change2_df])
    rate_df[stock['name']] = stock_df.cs

    info_df[stock['name']] = [
        get_stock_data.info['marketCap'], 
        get_kor_amount_string_no_change(get_stock_data.info['marketCap']),
        get_stock_data.info['recommendationKey'],
        get_stock_data.info['currentPrice'],
        '']

status_text.text("")
progress_bar.empty()
st.write(f""" ### 🚀 {dt_range} 누적변동률  """)

line_chart = alt.Chart(change_df).mark_line().encode(
    x = alt.X('Date:T', title=''),
    y = alt.Y('rate:Q', title=''),
    # color = alt.Color('symbol:N', title='종목', legend=None)
    color = alt.Color('symbol:N', title='', legend=alt.Legend(
        orient='bottom', #none
        # legendX=130, legendY=0,
        direction='horizontal',
        titleAnchor='end'))
)

text_data = change_df.loc[change_df['Date'].idxmax()]
text_data.reset_index(drop=True, inplace=True)
text_data2 = text_data.sort_values(by=['rate'], ascending=True)
text_data2.reset_index(drop=True, inplace=True)
text_data3 = pd.DataFrame(text_data.loc[0]).T
if len(text_data2.index) > 1:
    text_data3.loc[1] = text_data2.loc[0]
if len(text_data2.index) > 2:
    text_data3.loc[2] = text_data2.loc[round(len(text_data3.index)/2)]

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
    y = alt.Y('rate:Q', title='변동률'),
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
    y = alt.Y('rate:Q', title='변동률'),
    # y = 'rate:Q',
    text=alt.Text('symbol:N'),
    color = alt.Color('symbol:N', title='')
)
st.altair_chart(line_chart + labels + labels2, use_container_width=True)

df2 = info_df.T
st.write(""" ### 🎙️ 시가총액 """)
# st.write(f""" #### (대우건설: {df2['시가총액변환'][0]} ) """)
df2['종목명'] = df2.index
bar_chart = alt.Chart(df2, title='').mark_bar().encode(
                x = alt.X('시가총액:Q', title='', axis=alt.Axis(labels=False)),
                y = alt.Y('종목명:O', title=''),
                color = alt.Color('종목명:N', title='종목', legend=None)   
            )

bar_text = alt.Chart(df2).mark_text(
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
                text=alt.Text('시가총액변환:N')
            )
st.altair_chart(bar_chart + bar_text, use_container_width=True)
with st.expander("상세표 보기"):
    st.table(df2) # 시가총액, 현재가
    st.table(text_data) # 변동률

st.write(""" ### 🎙️ AI 동종사 비교 """)
# DataFrame 결과를 ChatCompletion messages에 넣기 위한 변환
messages = [{'role': 'system', 'content': '넌 대우건설 재무 분석가야'},
            {'role': 'assistant', 'content': '비교 분석해줘'}]

userq = '|회사명|시가총액|매수의견|현재가|' + '\n'
# DataFrame의 각 행을 ChatCompletion messages에 추가
for index, row in df2.iterrows():
    # if index == ' 대우건설':
    #     st.write(row)
    userq = userq + '|' + index + '|' + row['시가총액변환'] + '|' + row['매수의견'] + '|'
    userq = userq + str(round(row['현재가'])) + '|' + '\n'
# st.write(userq)
# print(userq)
user_message = {'role': 'user', 'content': f"{userq}"}
messages.extend([user_message])

userq = '|회사명|변동률|' + '\n'
# DataFrame의 각 행을 ChatCompletion messages에 추가
for index, row in text_data.iterrows():
    rate = round(row['rate'], 2)
    userq = userq +  '|' + row['symbol'] + '|' + f"{rate}" + '|' + '\n'
# st.write(userq)
# print(userq)
user_message = {'role': 'user', 'content': f"{userq}"}
messages.extend([user_message])

with st.spinner('Waiting for ChatGPT...'):
    get_respense = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages = messages,
        # stream=True,   
    )
    prompt = get_respense["choices"][0]["message"]["content"]
    # print(prompt)

st.success(f""" {prompt} """)
# st.write(f""" {prompt} """)

with st.expander("상세표 보기"):
    st.write(messages)

### 사이드바 종목 설정 #########################################################
products = [
    {'name': ' 원/달러', 'symbol': 'USDKRW=X'}
    ]

multi_products = st.sidebar.multiselect(
    "지표를 선택하세요",
    [
        "크루드오일 CL=F",
        "Gold GC=F",
        "S&P500 ^GSPC",
        "천연가스 LNG",
        "10년물 ^TNX",
        "DBC DBC",
        "BTC-USD BTC-USD",
        "달러인덱스 DX-Y.NYB"
        ],
    [ #초기 선택
        "크루드오일 CL=F",
        "Gold GC=F",
        "S&P500 ^GSPC",
        "천연가스 LNG",
        "10년물 ^TNX",
        "DBC DBC",
        "BTC-USD BTC-USD",
        "달러인덱스 DX-Y.NYB"
        ]
    )

for product in multi_products:
    words = product.split()
    products.append({'name': words[0], 'symbol': words[1]})

### 공통함수 ###############################################################
change_df = pd.DataFrame() # 변동률
last_df = pd.DataFrame() # 변동률

# with st.spinner(text="각종 지표 불러오는중..."):
# with st.spinner(text="각종 지표 불러오는중..."):    
progress_bar = st.progress(0)
status_text = st.empty()
# for product in products:
for idx, product in enumerate(products):

    l_rate = round(i / len(products) * 100)
    progress_bar.progress(l_rate)
    # status_text.text("%i%% Complete" % l_rate)
    status_text.text("지표정보를 불러오는 중입니다. %i%%" % l_rate)

    get_product_data = yf.Ticker(product['symbol'])
    product_df = get_product_data.history(period='1d', start=start_date, end=end_date)

    # 일간변동률, 누적합계
    product_df['dpc'] = (product_df.Close/product_df.Close.shift(1)-1)*100
    product_df['cs'] = product_df.dpc.cumsum()

    change2_df = pd.DataFrame(
        {
            'symbol': product['name'],
            # 'date': product_df.index,
            # 'idx': change2_df.index,
            # 'date_type': product_df.index,
            'Close': product_df.Close,
            'rate': product_df.cs,
            }
    )
    change2_df.reset_index(drop=False, inplace=True)
    change_df = pd.concat([change_df, change2_df])

    last2_df = pd.DataFrame(product_df.iloc[len(product_df.index)-1]).T
    last3_df = pd.DataFrame(
        {
            'symbol': product['name'],
            'Date': last2_df.index,
            'Close': last2_df.Close, 
            # 'idx': change2_df.index,
            # 'date_type': product_df.index,
            'rate': last2_df.cs,
            }
    )
    # st.write(last3_df)
    # last3_df.reset_index(drop=False, inplace=True)
    last_df = pd.concat([last_df, last3_df])
    # last3_df.reset_index(drop=False, inplace=True)
    # last_df.reset_index(drop=False, inplace=True)

status_text.text("")
progress_bar.empty()
st.write(f""" ### 📈 {dt_range} 지표변동률  """)

line_chart = alt.Chart(change_df).mark_line().encode(
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
    y = alt.Y('rate:Q', title='변동률'),
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
    y = alt.Y('rate:Q', title='변동률'),
    text=alt.Text('symbol:N'),
    color = alt.Color('symbol:N', title='')
)

st.altair_chart(line_chart + labels + labels2, use_container_width=True)
with st.expander("상세표 보기"):
    st.write(text_data2)
    st.write(last_df)
    st.table(last_df)
    st.write(change_df)
    st.table(change_df)
    






