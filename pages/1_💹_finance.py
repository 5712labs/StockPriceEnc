import yfinance as yf
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

### 페이지 기본 설정 ##########################################################
st.set_page_config(
    page_icon="🤣",
    page_title="5712labs FLiveAnalytics",
    layout="wide",
)
st.header("시계열분석 👋 ")

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
# end_date = st.sidebar.date_input('End date', datetime.today())
end_date = datetime.today()


# 페이지 컬럼 분할(예: 부트스트랩 컬럼, 그리드)
cols = st.columns((1, 1, 2))
cols[0].metric("10/11", "15 °C", "2")
cols[0].metric("10/12", "17 °C", "2 °F")
cols[0].metric("10/13", "15 °C", "2")
cols[1].metric("10/14", "17 °C", "2 °F")
cols[1].metric("10/15", "14 °C", "-3 °F")
cols[1].metric("10/16", "13 °C", "-1 °F")
# 라인 그래프 데이터 생성(with. Pandas)
chart_data = pd.DataFrame(
    np.random.randn(20, 3),
    columns=['a', 'b', 'c'])

# 컬럼 나머지 부분에 라인차트 생성
cols[2].line_chart(chart_data)

st.write(chart_data)


tickers_kr = {
    "047040.KS" : " 대우건설",
    "000720.KS" : "현대건설",
    "375500.KS" : "DL이앤씨",
    "006360.KS" : "GS건설",
    "028260.KS" : "삼성물산"
}

tickers = list(tickers_kr.keys())
get_stock_data = yf.download(tickers, start_date, end_date)
st.write(get_stock_data)
get_stock_data.rename(columns=tickers_kr, inplace=True)
st.write(get_stock_data)

# 종가 데이터 추출
close = get_stock_data.Close.copy()
# st.write(close)

# 정규환 하기
norm = close.div(close.iloc[0].mul(1))
# st.write(norm)
# stock_df = get_stock_data.history(period='1d', start=start_date, end=end_date)
st.line_chart(norm)

# arr = np.random.normal(1, 1, size=100)
# fig, ax = plt.subplots()
# ax.hist(arr, bins=20)
# st.write(arr)
# st.pyplot(fig)

# tickers_kr = {
#     "047040.KS" : " 대우건설"
# }

# get_stock_data = yf.download(tickers, start_date, end_date)
# st.write(get_stock_data)

# get_stock_data = yf.Ticker('047040.KS')
# stock_df = get_stock_data.history(period='1d', start=start_date, end=end_date)
get_stock_data = yf.Ticker('GOOG')
stock_df = get_stock_data.history(period='1d', start='2012-10-31', end='2022-10-31')

data = stock_df['Close'][stock_df['Volume'] != 0]
st.write(data)



# 원본시계열, 이동평균, 이동표준편차 시각화
# def plot_rolling(data, interval):
#     rolmean = data.rolling(interval).mean()
#     rolstd = data.rolling(interval).std()
#     # Plot rolling statistics:
#     fig = plt.figure(figsize=(10, 6))
#     plt.xlabel('Date')
#     orig = plt.plot(data, color='blue',label='Original')
#     mean = plt.plot(rolmean, color='red', label='Rolling Mean {}'.format(interval))
#     std = plt.plot(rolstd, color='black', label = 'Rolling Std {}'.format(interval))
#     plt.legend(loc='best')
#     plt.title('Rolling Mean & Standard Deviation')
#     # plt.show()
#     st.pyplot(fig)

# # 50일치 평균내어 이동평균계산
# plot_rolling(data, 50)

rolmean = data.rolling(50).mean()
rolstd = data.rolling(50).std()
# Plot rolling statistics:
fig = plt.figure(figsize=(10, 6))
plt.xlabel('Date')
orig = plt.plot(data, color='blue',label='Original')
mean = plt.plot(rolmean, color='red', label='Rolling Mean {}'.format(50))
std = plt.plot(rolstd, color='black', label = 'Rolling Std {}'.format(50))
plt.legend(loc='best')
plt.title('Rolling Mean & Standard Deviation')
# plt.show()
# st.pyplot(fig)

cols = st.columns((1, 1))
cols[0].pyplot(fig)
cols[1].metric("10/11", "15 °C", "2")


def adf_test(data):
    result = adfuller(data.values)
    print('ADF Statistics: %f' % result[0])
    print('p-value: %f' % result[1])
    print('num of lags: %f' % result[2])
    print('num of observations: %f' % result[3])
    print('Critical values:')
    for k, v in result[4].items():
        print('\t%s: %.3f' % (k,v))

print('ADF TEST 결과')
adf_test(data)

dff1 = data.diff().dropna()
fig = plt.figure(figsize=(10, 6))
dff1.plot(figsize=(15,5))
st.pyplot(fig)

# 차분 테이터 adf테스트
print('ADF TEST 결과')
adf_test(dff1)

fig = plot_acf(data)
st.pyplot(fig)
fig = plot_pacf(data)
st.pyplot(fig)

fig = plot_acf(dff1)
st.pyplot(fig)
fig = plot_pacf(dff1)
st.pyplot(fig)

