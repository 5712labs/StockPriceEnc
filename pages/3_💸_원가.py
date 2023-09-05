import streamlit as st
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
from pandas.tseries.offsets import MonthEnd, MonthBegin
import altair as alt
import convert

st.title('준공예정도급')

if convert.check_password() == False:
    st.stop()

# xlsx_row_df = pd.read_excel('./sources/base.xlsx')
xlsx_row_df = convert.get_df_from_password_excel('./sources/base.xlsx', st.secrets["password"])
xlsx_row_df['결재일'] = pd.to_datetime(xlsx_row_df['결재일'])
xlsx_row_df['공사종료일'] = pd.to_datetime(xlsx_row_df['공사종료일'])

xlsx_df = xlsx_row_df[['결재일', '현장코드', '현장명', '       준공예정원가율', '공사종료일', '           준공예정도급액', '          준공예정원가']]
xlsx_df.columns = ['결재일', '현장코드', '현장명', '준공예정원가율', '공사종료일', '준공예정도급액', '준공예정원가']

xlsx_df.drop(xlsx_df[(xlsx_df['공사종료일'] <= datetime.today())].index, inplace=True)
xlsx_df.reset_index(drop=True, inplace=True)
# st.write(f'xlsx_df{xlsx_df.shape}')
# st.write(xlsx_df)
xlsx_df['결재일'] = xlsx_df['결재일'] + MonthEnd()
xlsx_df.groupby(['현장코드'])['결재일'].apply(lambda x:pd.date_range(start=x.min(), end=datetime.today(), freq="M")).explode().reset_index().merge(xlsx_df, how='left').ffill()
xlsx_df = xlsx_df.groupby(['현장코드'])['결재일'].apply(lambda x:pd.date_range(start=x.min(), end=datetime.today(), freq="M")).explode().reset_index().merge(xlsx_df, how='left').ffill()

grouped_df  = xlsx_df.groupby('현장코드')

change_eco_df = pd.DataFrame() # 변동률
last_df = pd.DataFrame() # 변동률
for key ,data in grouped_df :
    sort_data = data.sort_values(by=['결재일'], ascending=True)
    sort_data['dpc'] = (sort_data.준공예정원가율/sort_data.준공예정원가율.shift(1)-1)*100
    sort_data['변동률'] = round(sort_data.dpc.cumsum(), 2)
    change_eco_df = pd.concat([change_eco_df, sort_data])

    last3_df = pd.DataFrame(change_eco_df.iloc[len(change_eco_df.index)-1]).T
    last_df = pd.concat([last_df, last3_df])

change_eco_df.drop(change_eco_df[(change_eco_df['변동률'] == 0)].index, inplace=True)
change_eco_df.dropna(inplace=True)

last_df.drop(last_df[(last_df['변동률'] == 0)].index, inplace=True)
last_df.dropna(inplace=True)
last_df.sort_values(by=['변동률'], ascending=False, inplace=True)

##########################################################################
### ####################################################
##########################################################################
st.write(""" ### 준공예정도급액 (단위: 천억)""")
bar_chart = alt.Chart(last_df).mark_bar().encode(
    # x = alt.X("준공예정도급액:Q", bin=True),
    # alt.X("준공예정도급액", bin=alt.Bin(maxbins=100000000)),
    # alt.X("준공예정도급액", bin=alt.Bin(step=10000000000)), #백억
    x = alt.X("준공예정도급액", bin=alt.Bin(step=100000000000)), #천억
    # x = alt.X('준공예정도급액:Q', axis=alt.Axis(format="%Y-%b-%d", labelOverlap=False, labelAngle=-45)),
    y = 'count(*):Q',
)
# ).properties(
#             height=180,
#             width=400,
#             ).interactive()
st.altair_chart(bar_chart, use_container_width=True)

##########################################################################
### ####################################################
##########################################################################
st.write(""" ### 준공예정도급액 분포 (단위: 백억)""")
bar_chart = alt.Chart(last_df).mark_bar().encode(
    # x = alt.X("준공예정도급액:Q", bin=True),
    # alt.X("준공예정도급액", bin=alt.Bin(maxbins=100000000)),
    alt.X("준공예정도급액:Q", bin=alt.Bin(step=10000000000)), #백억
    # x = alt.X("준공예정도급액", bin=alt.Bin(step=100000000000)), #천억
    # x = alt.X('준공예정도급액:Q', axis=alt.Axis(format="%Y-%b-%d", labelOverlap=False, labelAngle=-45)),
    y = 'count(*):Q',
)
# ).properties(
#             height=180,
#             width=400,
#             ).interactive()
st.altair_chart(bar_chart, use_container_width=True)


##########################################################################
### ####################################################
##########################################################################
st.write(""" ### 준공예정원가율 누적변동율 """)
lines = alt.Chart(change_eco_df).mark_line().encode(
    x = alt.X('결재일:T', title=''),
    # y = alt.Y('준공예정원가율:Q', title=''),
    y = alt.Y('변동률:Q', title=''),
    color = alt.Color('현장명:N', title='지표', legend=alt.Legend(
        orient='bottom', 
        direction='horizontal',
        titleAnchor='end'))
)

# st.write(last_df)
# text_data = last_df.sort_values(by=['변동률'], ascending=False).dropna()
text_data = last_df
text_data.reset_index(drop=True, inplace=True)
text_data3 = pd.DataFrame(text_data.loc[0].T)
text_data3 = text_data[:1]
if len(text_data.index) > 1:
    text_data3.loc[1] = text_data.loc[len(text_data.index)-1]
if len(text_data.index) > 2:
    text_data3.loc[2] = text_data.loc[round(len(text_data.index)/2)]

labels = alt.Chart(text_data3).mark_text(
    # point=True,
    fontWeight=600,
    fontSize=15,
    # color='white',
    align='left',
    dx=15,
    dy=-8
).encode(
    x = alt.X('결재일:T', title=''),
    # y = alt.Y('rate:Q', title=rate_text),
    y = alt.Y('변동률:Q', title=''),
    # y = 'rate:Q',
    text=alt.Text('변동률:Q', format='.1f'),
    # text=alt.TextValue(text_data3['현장명'][0] + text_data3['현장명'][0]),
    color = alt.Color('현장명:N', title='')
)

labels2 = alt.Chart(text_data3).mark_text(
    # point=True,
    fontWeight=600,
    fontSize=15,
    # color='white',
    align='left',
    dx=15,
    dy=10
).encode(
    x = alt.X('결재일:T', title=''),
    # y = alt.Y('rate:Q', title=rate_text),
    y = alt.Y('변동률:Q', title=''),
    # y = 'rate:Q',
    text=alt.Text('현장명:N'),
    # text=alt.TextValue(text_data3['현장명'][0] + text_data3['현장명'][0]),
    color = alt.Color('현장명:N', title='')
)

columns = sorted(change_eco_df.현장명.unique())
selection = alt.selection_point(
    fields=['결재일'], nearest=True, on='mouseover', empty=False, clear='mouseout'
)
points = lines.mark_point().transform_filter(selection)

base = alt.Chart(change_eco_df).encode(x='결재일:T')
rule = base.transform_pivot(
    '현장명', value='변동률', groupby=['결재일']
    ).mark_rule().encode(
    opacity=alt.condition(selection, alt.value(0.3), alt.value(0)),
    tooltip=[alt.Tooltip(c, type='quantitative') for c in columns]
).add_params(selection)

st.altair_chart(lines + rule + points + labels + labels2, 
                use_container_width=True)

##########################################################################
### ####################################################
##########################################################################
# st.write(""" ### 준공예정원가율 누적변동율""")
bar_chart = alt.Chart(change_eco_df, title='').mark_bar().encode(
                x = alt.X('현장명:O', title='', axis=alt.Axis(labels=False)),
                y = alt.Y('변동률:Q', title='')
                # color = alt.Color('종목명:N', title='종목', legend=None)   
            )
st.altair_chart(bar_chart, use_container_width=True)

##########################################################################
### ####################################################
##########################################################################
st.write(""" ### 준공예정원가율 분포""")
bar_chart = alt.Chart(change_eco_df).mark_bar().encode(
    alt.X("준공예정원가율:Q", bin=alt.Bin(step=5)), #백억
    y='count()',
)
st.altair_chart(bar_chart, use_container_width=True)

# lines = alt.Chart(change_eco_df).mark_line().encode(
#     x = alt.X('결재일:T', title=''),
#     # y = alt.Y('준공예정원가율:Q', title=''),
#     y = alt.Y('준공예정원가율:Q', title=''),
#     color = alt.Color('현장명:N', title='지표', legend=alt.Legend(
#         orient='bottom', 
#         direction='horizontal',
#         titleAnchor='end'))
# )
# st.altair_chart(lines, 
#                 use_container_width=True)


with st.expander("프롬프트 보기"):
    st.write(f"change_eco_df{change_eco_df.shape}")
    st.write(change_eco_df)
    st.write(f"last_df{last_df.shape}")
    st.write(last_df)
    st.write(f"xlsx_row_df{xlsx_row_df.shape}")
    st.write(xlsx_row_df)
    
