import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns

st.header("Mobile Prediction project")

data = pd.read_csv('train.csv',)
st.write(data.head())


st.write (data.isna().sum())

st.write (data.columns)



fig, axs = plt.subplots(22, figsize = (10,25))
colors = ['#0000FF', '#00FF00',
          '#FFFF00', '#FF00FF']


st.pyplot(plt1 = sns.boxplot(data['battery_power'], ax = axs[0]))
st.pyplot(plt2 = sns.boxplot(data['blue'], ax = axs[1]) )
st.pyplot(plt3 = sns.boxplot(data['clock_speed'], ax = axs[2]) )
st.pyplot(plt4 = sns.boxplot(data['dual_sim'], ax = axs[3]) )
st.pyplot(plt5 = sns.boxplot(data['fc'], ax = axs[4]) )
st.pyplot(plt6 = sns.boxplot(data['four_g'], ax = axs[5]))
st.pyplot(plt7 = sns.boxplot(data['int_memory'], ax = axs[6]) )
st.pyplot(plt8 = sns.boxplot(data['m_dep'], ax = axs[7]) )
st.pyplot(plt9 = sns.boxplot(data['mobile_wt'], ax = axs[8]) )
st.pyplot(plt10 = sns.boxplot(data['n_cores'], ax = axs[9]))
st.pyplot(plt11 = sns.boxplot(data['pc'], ax = axs[10]) )
st.pyplot(plt12 = sns.boxplot(data['px_height'], ax = axs[11]) )
st.pyplot(plt13 = sns.boxplot(data['px_width'], ax = axs[12]) )
st.pyplot(plt14 = sns.boxplot(data['ram'], ax = axs[13]) )
st.pyplot(plt15 = sns.boxplot(data['four_g'], ax = axs[14])  )
st.pyplot(plt16 = sns.boxplot(data['int_memory'], ax = axs[15]))
st.pyplot(plt17 = sns.boxplot(data['sc_h'], ax = axs[16]) )
st.pyplot(plt18 = sns.boxplot(data['talk_time'], ax = axs[17]) )
st.pyplot(plt19= sns.boxplot(data['three_g'], ax = axs[18]))
st.pyplot(plt20 = sns.boxplot(data['touch_screen'], ax = axs[19]) )
st.pyplot(plt21 = sns.boxplot(data['wifi'], ax = axs[20])  )
st.pyplot(plt22 = sns.boxplot(data['price_range'], ax = axs[21]))

st.write(plt.tight_layout())


st.write(data.info())


