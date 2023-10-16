import streamlit as st
import pygwalker as pyg
import pandas as pd
import streamlit.components.v1 as components
from components import convert

st.header("EDA 👋 ")

if convert.check_password() == False:
    st.stop()

# 데이터 가져오기
# df = pd.read_csv("https://kanaries-app.s3.ap-northeast-1.amazonaws.com/public-datasets/bike_sharing_dc.csv")
xlsx_row_df = convert.get_df_from_password_excel('./sources/base.xlsx', st.secrets["password"])
# Pygwalker를 사용하여 HTML 생성
pyg_html = pyg.walk(xlsx_row_df, return_html=True)
 
# Streamlit 앱에 HTML 임베드
components.html(pyg_html, height=1000, scrolling=True)
