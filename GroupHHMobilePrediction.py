import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.metrics import confusion_matrix, classification_report

st.header("Mobile Prediction project")

data = pd.read_csv(r'train.csv',)

if st.checkbox('Show Dataframe'):
  st.write(data)
  
  st.write('This is a column.')
  st.write (data.columns)

st.write('This is a pie chart for price range.')

pie_chart = px.pie(data,"price_range")
st.plotly_chart(pie_chart)

#st.write('This is a outlier for px_height.')
#fig,ax = plt.subplots()
#ax.box(data['px_height'],bins=20)
#st.pyplot(fig)

st.write('To see correlation')
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score

dcopy=data.copy()

dcopy_new=dcopy

dcopy_new[['clock_speed', 'm_dep','fc','px_height']] = dcopy[['clock_speed', 'm_dep','fc','px_height']].astype('int64') 

#matrix = dcopy.corr()
#f, ax = plt.subplots(figsize=(20, 15))
#sns.heatmap(matrix, vmax=1, square=True, annot=True,cmap='Paired')

#fig, ax = plt.subplots()
#sns.heatmap(matrix, ax=ax)
#st.pyplot(fig)

if st.checkbox("Show Correlation Plot"):
            st.write("### Heatmap")
            fig, ax = plt.subplots(figsize=(30,15))
            st.write(sns.heatmap(dcopy.corr(), annot=True,linewidths=0.7,cmap='Set3'))# Train the model
            st.pyplot()
          

from sklearn.model_selection import train_test_split
X=dcopy_new.drop('price_range',axis=1)
y=dcopy_new['price_range']


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

useless_col = ['battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc', 'four_g',
       'int_memory', 'm_dep', 'mobile_wt', 'n_cores', 'pc', 'px_height',
       'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time', 'three_g',
       'touch_screen', 'wifi']

data_modelling = dcopy_new.drop(useless_col, axis = 1)

y = data_modelling['price_range']
X1 = data_modelling.drop('price_range', axis = 1)
X2 = pd.get_dummies(data_modelling)
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1,y,random_state=42,test_size=0.2)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2,y,random_state=42,test_size=0.2)

        
        
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix, classification_report

# Logistic Regression

# without reduction
logregwithoutpca = LogisticRegression()
logregwithoutpca.fit(X_train, y_train)

logregwithoutpca_result = logregwithoutpca.predict(X_test)#After training-need to perdict

st.write('Accuracy of Logistic Regression (without PCA) on training set: {:.2f}'
     .format(logregwithoutpca.score(X_train, y_train)))
st.write('Accuracy of Logistic Regression (without PCA)  on testing set: {:.2f}'
     .format(logregwithoutpca.score(X_test, y_test)))
st.write('\nConfusion matrix :\n',confusion_matrix(y_test, logregwithoutpca_result))
#print('\n\nClassification report :\n\n', classification_report(y_test, logregwithoutpca_result))

#print

#ConfusionMatrix 

st.write("Visualization Confusion Matrix")
confusion_matrix =confusion_matrix(y_test, logregwithoutpca_result)
sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap='Set3')
plt.title('Confusion Matrix for KNN')
plt.xlabel('Predicted')
plt.ylabel('True')
st.write(classification_report(y_test, logregwithoutpca_result))
st.pyplot()


X=dcopy.drop(['price_range'],axis=1)
y=dcopy[['price_range']]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=101)




st.write("ConfusionMatrix In Percentage")
sns.heatmap(confusion_matrix/np.sum(confusion_matrix), annot=True, 
            fmt='.1%', cmap='Accent')
plt.title('Confusion Matrix for KNN In Percentage')
plt.xlabel('Predicted Value')
plt.ylabel('True')
st.write(classification_report(y_test, logregwithoutpca_result))

fig, ax = plt.subplots()

sns.heatmap(confusion_matrix, ax=ax)
st.write(classification_report(y_test, logregwithoutpca_result))
st.pyplot(fig
