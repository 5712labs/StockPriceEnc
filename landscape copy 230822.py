import yfinance as yf
import streamlit as st
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
import time
import altair as alt

### 페이지 기본 설정 ##########################################################
st.set_page_config(
    page_icon="🤣",
    page_title="5712labs FLiveAnalytics",
    # layout="wide",
)

# 페이지 헤더, 서브헤더 제목 설정
st.header("경제동향 👋 ")
# st.subheader("스트림릿 기능 맛보기")

### 사이드바 기간 설정 #########################################################
st.sidebar.header('Menu')

dt_range = st.sidebar.radio('기간', ['3개월', '6개월', '1년', '3년', '10년'])

if dt_range == '3개월':
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
products = [
    {'name': ' 원/달러', 'symbol': 'KRW=X'},
    # {'name': ' 애플', 'symbol': 'AAPL'},
    # {'name': ' 코스피', 'symbol': '^KS11'},
    # {'name': 'GS건설', 'symbol': '006360.KS'},
    # {'name': '현대건설', 'symbol': '000720.KS'},
    # {'name': 'DL이앤씨', 'symbol': '375500.KS'},
    # {'name': '삼성엔지니어링', 'symbol': '028050.KS'},
    # {'name': '금호건설', 'symbol': '002990.KS'},
    ]

multi_products = st.sidebar.multiselect(
    "동종사를 선택하세요",
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
        # "S&P500 ^GSPC",
        "천연가스 LNG",
        # "10년물 ^TNX",
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

with st.spinner(text="페이지 로딩중..."):
    # for product in products:
    for idx, product in enumerate(products):
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


st.write(""" ### 🚀  누적변동률 """)

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
    fontSize=13,
    # color='white',
    align='left',
    dx=15,
    dy=10
).encode(
    x = alt.X('Date:T', title=''),
    y = alt.Y('rate:Q', title='변동률'),
    # y = 'rate:Q',
    text=alt.Text('symbol:N'),
    color = alt.Color('symbol:N', title='')
)

st.altair_chart(line_chart + labels + labels2, use_container_width=True)
# with st.expander("상세표 보기"):
#     st.table(text_data2)
