import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.header("Mobile Prediction project")

data = pd.read_csv(r'train.csv',)
st.write(data.head())

st.write (data.isna().sum())
st.write (data.columns)


fig, ax = plt.subplots()
ax.box(data['battery_power'], bins=20)

st.pyplot(fig)
