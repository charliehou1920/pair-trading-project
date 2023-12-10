import streamlit as st
from metaflow import Flow
from metaflow import get_metadata, metadata
import pandas as pd

# make sure we point the app to the flow folder
# NOTE: MAKE SURE TO RUN THE FLOW AT LEAST ONE BEFORE THIS ;-)
FLOW_NAME = 'PairTrading' # name of the target class
# Set the metadata provider as the src folder in the project,
# which should contains /.metaflow
metadata('../trading')
# Fetch currently configured metadata provider - check it's local!
print(get_metadata())
    
@st.cache_resource
def get_latest_successful_run(flow_name: str):
    "Gets the latest successfull run."
    for r in Flow(flow_name).runs():

        if r.successful: 
            return r

last_run = get_latest_successful_run(FLOW_NAME)
results = last_run.data.all_results

def main():
    # Title of the website
    st.markdown("<h2 style='color: pink;'>Pair Trading with IT stocks in S&P500</h2>", unsafe_allow_html=True)

    # Image showing the clustering result
    st.image("../image/OPTICS_Clustering.png")

    # Brief Summary of The Project
    st.write("We use OPTICS and LSTM algorithms for stock pairing and trading.")

    # 提取股票对用于下拉框
    pairs = [result['pair'] for result in results]

    # 创建一个下拉框让用户选择股票对
    selected_pair = st.selectbox("Select a stock pair:", pairs)

    # 找到选中的股票对对应的数据
    selected_result = next((item for item in results if item['pair'] == selected_pair), None)

    # 如果找到了选中的股票对的数据
    if selected_result:
        # 将结果转换为DataFrame以便于显示
        df = pd.DataFrame.from_dict(selected_result['results'], orient='index', columns=['Value'])
        # 显示表格
        st.table(df)

if __name__ == "__main__":
    main()
