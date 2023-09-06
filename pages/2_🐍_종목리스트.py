# Import libraries
# https://sjblog1.tistory.com/41
# http://data.krx.co.kr/contents/MDC/MDI/mdiLoader/index.cmd?menuId=MDC0301
import streamlit as st
import pandas as pd
import numpy as np
from pykrx import stock
import FinanceDataReader as fdr
from datetime import datetime
from dateutil.relativedelta import relativedelta

# Page setup
st.set_page_config(page_title="Python Talks Search Engine", page_icon="🐍", layout="wide")
st.title("Stock Search Engine")

clear_button = st.sidebar.button("Clear Cache", key="clear")
if clear_button:
    st.cache_data.clear()

# krx_url = 'https://kind.krx.co.kr/corpgeneral/corpList.do?method=download&searchType=13'
# fdr_datastk_data = pd.read_html(krx_url, header=0)[0]  # 해당 site에서 table 추출 및 header는 가장 첫번째 행
# st.write(f'fdr_datastk_data {fdr_datastk_data.shape}')
# st.write(fdr_datastk_data)
# stk_data = fdr_datastk_data[['종목코드', '업종', '주요제품']]
# # 종목코드가 모두 6자리로 이루어져있지만 혹시 모르니, 6자리 미만 코드는 앞에 0을 채워넣어 6자리 숫자텍스트로 변환
# stk_data['종목코드'] = stk_data['종목코드'].apply(lambda input: '0' * (6 - len(str(input))) + str(input))
# st.write(f'stk_data {stk_data.shape}')
# st.write(stk_data)

# fdr_data = fdr.StockListing("KRX")
# st.write(f'fdr_data {fdr_data.shape}')
# st.write(fdr_data)

# stocks = stock.get_market_fundamental_by_ticker(date=datetime.today(), market="ALL")
# st.write(f'stocks {stocks.shape}')
# st.write(stocks)

# stocks_KS22 = pd.DataFrame({'종목코드':stock.get_market_ticker_list(market="KOSDAQ")})
# stocks_KS22['종목명'] = stocks_KS22['종목코드'].map(lambda x: stock.get_market_ticker_name(x))
# st.write(stocks_KS22)


# stocks_KQ22 = pd.DataFrame({'종목코드':stock.get_market_ticker_list(market="KOSPI")})
# stocks_KQ22['종목명'] = stocks_KQ22['종목코드'].map(lambda x: stock.get_market_ticker_name(x))
# st.write(stocks_KQ22)

# @st.cache_data
def load_data():
    stocks_KQ = pd.DataFrame({'종목코드':stock.get_market_ticker_list(market="KOSPI")})
    stocks_KQ['종목명'] = stocks_KQ['종목코드'].map(lambda x: stock.get_market_ticker_name(x))
    stocks_KQ['코드'] = stocks_KQ['종목코드'] + '.KS'
    
    stocks_KS = pd.DataFrame({'종목코드':stock.get_market_ticker_list(market="KOSDAQ")})
    stocks_KS['종목명'] = stocks_KS['종목코드'].map(lambda x: stock.get_market_ticker_name(x))
    stocks_KS['코드'] = stocks_KS['종목코드'] + '.KQ'

    stocks_KQKS = pd.concat([stocks_KQ, stocks_KS])
    
    # 전체 종목의 시가총액 
    # fdr_data = fdr.StockListing("KRX")
    # stocks_KQ['시총'] = stocks_KQ['종목명'].map(lambda x: fdr_data[fdr_data["Name"]==x]["Marcap"].iloc[0])
    
    # 전체 종목의 업종 
    krx_url = 'https://kind.krx.co.kr/corpgeneral/corpList.do?method=download&searchType=13'
    fdr_datastk_data = pd.read_html(krx_url, header=0, encoding='cp949')[0]  # 해당 site에서 table 추출 및 header는 가장 첫번째 행
    stk_data = fdr_datastk_data[['종목코드', '업종', '주요제품']]     # 9개의 열 중 '회사명', '종목코드' 만 추출하여 dataframe 완성
    # 종목코드가 모두 6자리로 이루어져있지만 혹시 모르니, 6자리 미만 코드는 앞에 0을 채워넣어 6자리 숫자텍스트로 변환
    stk_data['종목코드'] = stk_data['종목코드'].apply(lambda input: '0' * (6 - len(str(input))) + str(input))
    stock_list = pd.merge(stocks_KQKS, stk_data, on='종목코드', how='left')
    # 전체 종목의 펀더멘탈 지표 가져오기
    # 펀더멘탈 지표는 PER, PBR, EPS, BPS, DIV, DPS를 가져옵니다.
    search_date = datetime.today() - relativedelta(day=1)
    stock_fud = pd.DataFrame(stock.get_market_fundamental_by_ticker(date=search_date, market="ALL"))
    stock_fud = stock_fud.reset_index()
    stock_fud.rename(columns={'티커':'종목코드'}, inplace=True)

    # result = pd.merge(stock_list, stock_fud, left_on='종목코드', right_on='종목코드', how='left')
    result = pd.merge(stock_list, stock_fud, on='종목코드', how='left')
    
    stock_price = stock.get_market_ohlcv_by_ticker(date=search_date, market="ALL")
    stock_price = stock_price.reset_index()
    stock_price.rename(columns={'티커':'종목코드'}, inplace=True)
    # result1 = pd.merge(result, stock_price, left_on='종목코드', right_on='종목코드', how='left')
    result1 = pd.merge(result, stock_price, on='종목코드', how='left')

    #코넥스 제거
    result1.dropna(subset=['종목명'], how='any', axis=0, inplace=True)
    
    # result1 = result1.replace([0], np.nan)    # 0값을 NaN으로 변경
    # result1 = result1.dropna(axis=0)      # NaN을 가진 행 제거
    result1 = result1.sort_values(by=['PER'], ascending=True)
    result1['내재가치'] = (result1['BPS'] + (result1['EPS']) * 10) / 2
    result1['내재가치/종가'] = (result1['내재가치'] / result1['종가'])
    # st.write('result1')
    # st.write(result1.head())
    return result1

analy = load_data()

with st.expander(f"전체 보기{analy.shape}"):
    st.write(analy)

text_search = st.text_input("종목코드, 종목명, 업종, 주요제품으로 검색하세요", value="")
# Filter the dataframe using masks
m1 = analy["종목코드"].str.contains(text_search)
m2 = analy["종목명"].str.contains(text_search)
m3 = analy["업종"].str.contains(text_search)
m4 = analy["주요제품"].str.contains(text_search)
# if text_search.contains('PER'):
# if text_search in 'PER':
#     m5 = analy["PER"] > 10
# else:
#     m5 = analy["PER"] < 10

# df_search = analy[m1 | m2 | m3 | m4 | m5]
df_search = analy[m1 | m2 | m3 | m4]
# df_search = analy[m3 | m4]
if text_search:
    st.write(df_search)

# Another way to show the filtered results
# Show the cards
N_cards_per_row = 6
if text_search:
    for n_row, row in df_search.reset_index().iterrows():
        i = n_row%N_cards_per_row
        if i==0:
            st.write("---")
            cols = st.columns(N_cards_per_row, gap="small")
        # draw the card
        with cols[n_row%N_cards_per_row]:
            # st.button()
            # st.caption(f"{row['Evento'].strip()} - {row['Lugar'].strip()} - {row['Fecha'].strip()} ")
            st.button(f"**{row['종목명']}**", type="secondary")
            # clear_button = st.sidebar.button("Clear Cache", key="clear")
            # if clear_button:
            #     st.cache_data.clear()
            st.caption(f"{row['코드']}")
            st.markdown(f"****{row['주요제품']}****")
            st.markdown(f"**{row['종목코드']}**")
            # st.markdown(f"*{row['종목명'].strip()}*")

# if st.button("Confirm"):
#     con = st.container()
#     con.caption("Result")
    # con.write(f"Hello~ {str(input_user_name)}")

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