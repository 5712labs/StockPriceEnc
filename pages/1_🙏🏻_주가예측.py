import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from pandas_datareader import data as pdr
import yfinance as yf
import streamlit as st
from datetime import datetime

st.header("대우건설 주가예측 🙏🏻")

# time = np.linspace(0, 1, 365*2)
# # result = np.sin(2*np.pi*12*time)
# result = np.sin(2*np.pi*12*time) + time + np.random.randn(365*2)/4
# ds = pd.date_range("2018-01-01", periods=365*2, freq="D")
# df = pd.DataFrame({"ds":ds, "y":result})

# fig = plt.figure(figsize=(10, 6))
# df["y"].plot(figsize=(10,6))
# st.pyplot(fig)

# # 학습
# m = Prophet(yearly_seasonality=True, daily_seasonality=True)
# m.fit(df)

# # 이후 30일간의 데이터 예측
# future = m.make_future_dataframe(periods=30)
# forecast = m.predict(future)

# fig = plt.figure(figsize=(10, 6))
# fig = m.plot(forecast)
# st.pyplot(fig)

########################################################################
########################################################################
########################################################################
start_date = "2010-03-01"
end_date = datetime.today()
# get_data_yahoo(종목코드, 시작일, 마감일)
# KIA = pdr.get_data_yahoo("000270.KS", start_date, end_date)
stock = pdr.get_data_yahoo("047040.KS", start_date, end_date)

# accuracy 확인을 위한 데이터
# stock_trunc = stock[:"2023-07-30"]
stock_trunc = stock[:-30]

df = pd.DataFrame({"ds":stock_trunc.index, "y":stock_trunc["Close"]})
# df = pd.DataFrame({"ds":KIA.index, "y":KIA["Close"]})
df.reset_index(inplace=True)
del df ["Date"]

m = Prophet(yearly_seasonality=True, daily_seasonality=True)
m.fit(df)

future = m.make_future_dataframe(periods=90, freq='D')
forecast = m.predict(future)
# st.write(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail())
forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail()

st.write(""" ### 10년 추세 """)
st.write(""" ##### ( 파란선 추세 / 주황선 예측 ) """)
# 예측 그래프
m.plot(forecast)
# component 별 그래프 (이미지 생략)
m.plot_components(forecast)

# fig = plt.figure(figsize=(10, 6))
fig = plt.figure(figsize=(8, 3))
fig = m.plot(forecast)
st.pyplot(fig)

fig2 = plt.figure(figsize=(8,3))
plt.plot(stock.index, stock["Close"], label="real")
plt.plot(forecast["ds"], forecast["yhat"], label="forecast")
st.pyplot(fig2)

st.write(""" ### 향후 3개월 예측 """)

# forecast = forecast[-10:] #마지막 10개?
forecast = forecast[3200:] #마지막 10개?
stock = stock[3200:]

# 예측 그래프
m.plot(forecast)
# component 별 그래프 (이미지 생략)
m.plot_components(forecast)

fig2 = plt.figure(figsize=(8,3))
plt.plot(stock.index, stock["Close"], label="real")
plt.plot(forecast["ds"], forecast["yhat"], label="forecast")
st.pyplot(fig2)

with st.expander("예측데이터 보기"):
  st.write(stock.tail())
  st.write(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]][-90:])
  




# yf.pdr_override()
# # 미국 실업률 데이터 불러오기
# start_date='2000-01-01'
# end_date ='2022-01-01'
# unrate = pdr.get_data_fred('UNRATE', start=start_date, end=end_date)
# # Prophet 모델에 넣을 수 있도록 데이터 변환
# unrate = unrate.reset_index()
# unrate = unrate.rename(columns={'DATE': 'ds', 'UNRATE': 'y'})
# # 모델 생성 및 학습
# model = Prophet()
# model.fit (unrate)
# # 1년 뒤 예측
# future = model.make_future_dataframe(periods=865)
# forecast = model.predict (future)
# # 결과 시각화
# model.plot(forecast, xlabel='Date', ylabel='Unemployment Rate (%)') 

# fig = plt.figure(figsize=(10, 6))
# plt.title('U.S. Unemployment Rate Forecast')
# # 트렌드 변화 시각화
# fig = model.plot_components(forecast)
# # plt.show()
# st.pyplot(fig)





