# Import libraries
# https://sjblog1.tistory.com/41
import streamlit as st
import pandas as pd
import numpy as np
from pykrx import stock
from datetime import datetime

# Page setup
st.set_page_config(page_title="Python Talks Search Engine", page_icon="🐍", layout="wide")
st.title("Stock Search Engine")

@st.cache_data
def load_data():
    stocks_KQ = pd.DataFrame({'종목코드':stock.get_market_ticker_list(market="KOSPI")})
    stocks_KQ['종목명'] = stocks_KQ['종목코드'].map(lambda x: stock.get_market_ticker_name(x))
    stocks_KQ['코드'] = stocks_KQ['종목코드'] + '.KS'
    # st.write(stocks_KQ.shape)
    # st.write(stocks_KQ.head())

    stocks_KS = pd.DataFrame({'종목코드':stock.get_market_ticker_list(market="KOSDAQ")})
    stocks_KS['종목명'] = stocks_KS['종목코드'].map(lambda x: stock.get_market_ticker_name(x))
    stocks_KS['코드'] = stocks_KS['종목코드'] + '.KQ'
    # st.write(stocks_KS.shape)
    # st.write(stocks_KS.head())

    stock_list = pd.concat([stocks_KQ, stocks_KS])
    # st.write(stock_list.shape)
    # st.write(stock_list.head())

    # 전체 종목의 펀더멘탈 지표 가져오기
    # 펀더멘탈 지표는 PER, PBR, EPS, BPS, DIV, DPS를 가져옵니다.
    stock_fud = pd.DataFrame(stock.get_market_fundamental_by_ticker(date=datetime.today(), market="ALL"))
    stock_fud = stock_fud.reset_index()
    stock_fud.rename(columns={'티커':'종목코드'}, inplace=True)
    result = pd.merge(stock_list, stock_fud, left_on='종목코드', right_on='종목코드', how='outer')
    # st.write('result')
    # st.write(result.head())

    stock_price = stock.get_market_ohlcv_by_ticker(date=datetime.today(), market="ALL")
    stock_price = stock_price.reset_index()
    stock_price.rename(columns={'티커':'종목코드'}, inplace=True)
    result1 = pd.merge(result, stock_price, left_on='종목코드', right_on='종목코드', how='outer')
    # result1 = result1.replace([0], np.nan)    # 0값을 NaN으로 변경
    # result1 = result1.dropna(axis=0)      # NaN을 가진 행 제거
    result1 = result1.sort_values(by=['PER'], ascending=True)
    result1['내재가치'] = (result1['BPS'] + (result1['EPS']) * 10) / 2
    result1['내재가치/종가'] = (result1['내재가치'] / result1['종가'])
    # st.write('result1')
    # st.write(result1.head())
    return result1

analy = load_data()
st.write(analy)

text_search = st.text_input("Search videos by title or speaker", value="")
# Filter the dataframe using masks
m1 = analy["종목코드"].str.contains(text_search)
m2 = analy["종목명"].str.contains(text_search)
df_search = analy[m1 | m2]
# Show the results, if you have a text_search
if text_search:
    st.write(df_search)

# Another way to show the filtered results
# Show the cards
# N_cards_per_row = 3
# if text_search:
#     for n_row, row in df_search.reset_index().iterrows():
#         i = n_row%N_cards_per_row
#         if i==0:
#             st.write("---")
#             cols = st.columns(N_cards_per_row, gap="large")
        # draw the card
        # with cols[n_row%N_cards_per_row]:
            # st.caption(f"{row['Evento'].strip()} - {row['Lugar'].strip()} - {row['Fecha'].strip()} ")
            # st.markdown(f"**{row['종목코드'].strip()}**")
            # st.markdown(f"*{row['종목명'].strip()}*")

st.stop()



# Connect to the Google Sheet
sheet_id = "1nctiWcQFaB5UlIs6z8d1O6ZgMHFDMAoo3twVxYnBUws"
sheet_name = "charlas"
url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
df = pd.read_csv(url, dtype=str).fillna("")

# Show the dataframe (we'll delete this later)
st.write(df)

text_search = st.text_input("Search videos by title or speaker", value="")
# Filter the dataframe using masks
m1 = df["Autor"].str.contains(text_search)
m2 = df["Título"].str.contains(text_search)
df_search = df[m1 | m2]
# Show the results, if you have a text_search
if text_search:
    st.write(df_search)
# Another way to show the filtered results
# Show the cards
N_cards_per_row = 3
if text_search:
    for n_row, row in df_search.reset_index().iterrows():
        i = n_row%N_cards_per_row
        if i==0:
            st.write("---")
            cols = st.columns(N_cards_per_row, gap="large")
        # draw the card
        with cols[n_row%N_cards_per_row]:
            st.caption(f"{row['Evento'].strip()} - {row['Lugar'].strip()} - {row['Fecha'].strip()} ")
            st.markdown(f"**{row['Autor'].strip()}**")
            st.markdown(f"*{row['Título'].strip()}*")
            st.markdown(f"**{row['Video']}**")