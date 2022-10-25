import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
st.set_option('deprecation.showPyplotGlobalUse', False)
from sklearn import datasets


# Diaplay Images
# import Image from pillow to open images
#from PIL import Image
#img = Image.open("MPhonePerdiction_Pics.jpg")
# display image using streamlit
# width is used to set the width of an image
#st.image(img, width=200)

st.write("""
# Simple Mobile Phone Price Prediction App
This app predicts the **MPhonePriceRange** type!
""")


##def main():
st.title("Mobile Phone Perdiction Price Automation")
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    st.write(bytes_data)

    # To convert to a string based IO:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    st.write(stringio)

    # To read file as string:
    string_data = stringio.read()
    st.write(string_data)

    # Can be used wherever a "file-like" object is accepted:
    dataframe = pd.read_csv(uploaded_file)
    st.write(dataframe)
    
def user_input_features():
 
   battery_power = st.sidebar.slider('Battery Power',0,800,2000)
   fc = st.sidebar.slider('Front Camera', 0,1,4)
   px_height = st.sidebar.slider('Phone Height',0,200,1000)
   px_width = st.sidebar.slider('Pixel Width',0,600,1750)
   data = {'BatteryCapacity': battery_power,
            'Front Camera': fc,
            'Phone Height': px_height,
            'Pixel Width': px_width}
   features = pd.DataFrame(data, index=[0])
   return features
  
## Side bar
st.sidebar.title("Mobile Phone Perdiction Price")
st.sidebar.header("Mobile Phone Features:")
dataframe = user_input_features()

st.subheader('User Input parameters')

data = st.file_uploader("Upload Dataset", type=['csv','txt',])

df = pd.DataFrame()
if data is not None:
   df = pd.read_csv(data)
   st.success("Data File Uploaded Successfully")
     
st.header("Mobile Phone Perdiction Price Automation")

data = pd.read_csv(r'train.csv',)
##data= pd.read_csv(uploaded_file)
st.write(data)

if st.checkbox('Show Dataframe'):
    st.write(data)
    st.write('This is a column.')
    st.write (data.columns)
 
st.write('This is a pie chart for price range.')

pie_chart = px.pie(data,"price_range")
st.plotly_chart(pie_chart)

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score
dcopy=data.copy()
dcopy_new=dcopy
dcopy_new[['clock_speed', 'm_dep','fc','px_height']] = dcopy[['clock_speed', 'm_dep','fc','px_height']].astype('int64') 


if st.checkbox('Show Mobile Correlation Plot'):
   st.write("Mobile Price-Perdiction-Correlation Between The Features")
   st.write("Correlation Mobile Price is highly correlated with RAM(Phone Memory)")
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
st.write('\n\nClassification report :\n\n', classification_report(y_test, logregwithoutpca_result))

#ConfusionMatrix 

confusion_matrix =confusion_matrix(y_test, logregwithoutpca_result)

st.write("Visualization Confusion Matrix")
plt.figure(figsize=(10,10))          
sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap='Set3')
plt.title('Confusion Matrix for KNN')
plt.xlabel('Predicted')
plt.ylabel('True')
st.write(classification_report(y_test, logregwithoutpca_result))
st.pyplot()


st.write("ConfusionMatrix In Percentage")
plt.figure(figsize=(10,10))          
sns.heatmap(confusion_matrix/np.sum(confusion_matrix), annot=True,fmt='.1%', cmap='Set3')
plt.title('Confusion Matrix for KNN In Percentage')
plt.xlabel('Predicted Value')
plt.ylabel('True')
st.write(classification_report(y_test, logregwithoutpca_result))
##fig, ax = plt.subplots()
st.pyplot()


st.write("ConfusionMatrix Actual Value and Percentage")
plt.figure(figsize=(10,10)) 
group_counts = ["{0:0.0f}".format(value) for value in
                confusion_matrix.flatten()]
group_percentages = ["{0:.1%}".format(value) for value in
                     confusion_matrix.flatten()/np.sum(confusion_matrix)]
labels = [f"{v2}\n{v3}" for v2, v3 in
          zip(group_counts,group_percentages)]
labels = np.asarray(labels).reshape(4,4)
sns.heatmap(confusion_matrix, annot=labels, fmt='', cmap='Set3')
plt.title('Actual Value and Percentage Plots ')
st.pyplot()

plt.clf()

plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Pastel2)
plt.figure(figsize=(10,10)) 
classNames = ['Negative','Positive','Positive','Positive']
plt.title('Mobile Phone Perdiction Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted ')
tick_marks = np.arange(len(classNames))
plt.xticks(tick_marks, classNames, rotation=90)
plt.yticks(tick_marks, classNames)
s = [['TN','FP','TP','TP'], ['TN','FP','TP','TP'],['TN','FP','TP','TP'],['TN','FP','TP','TP']]
for i in range(4):
    for j in range(4):
        plt.text(j,i, str(s[i][j])+" = "+str(confusion_matrix[i][j]))
plt.show()
st.pyplot()


st.write("Data Visualisation:")
st.write(sns.jointplot(x='ram',y='price_range',data=dcopy_new,color='brown',kind='kde'))
st.pyplot()

st.write(sns.pointplot(y="int_memory", x="price_range", data=dcopy_new))
st.pyplot()
