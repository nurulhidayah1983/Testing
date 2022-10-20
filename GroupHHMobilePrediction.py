import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from statistics import mean

st.header("Mobile Prediction project")

data = pd.read_csv(r'train.csv',)

if st.checkbox('Show Dataframe for Mobile Perdictions'):
  st.write(data)
  st.write('There are columns for the above Datasets.')
  st.write (data.columns)
st.write('This is a pie chart for price range and showing balanced Dataset.')

pie_chart = px.pie(data,"price_range")
st.plotly_chart(pie_chart)
st.write('This is a summary of the MobileData Perdictions.')

data.info()
dcopy=data.copy()
st.write(dcopy.describe())


st.write('The heatmap to show the most correlated between the features/columns and target')
matrix = dcopy.corr()
f, ax = plt.subplots(figsize=(40, 20))
sns.heatmap(matrix, vmax=1, square=True, annot=True,cmap='Paired')

fig, ax = plt.subplots()
sns.heatmap(matrix, ax=ax)
st.write(fig)


