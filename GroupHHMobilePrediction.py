import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

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

matrix = dcopy.corr()
f, ax = plt.subplots(figsize=(20, 15))
sns.heatmap(matrix, vmax=1, square=True, annot=True,cmap='Paired')

fig, ax = plt.subplots()
sns.heatmap(matrix, ax=ax)
st.pyplot(fig)

st.write("Conclusion: The most correlated features are:price_range vs Ram has correlation coefficient of 0.92.")

useless_col = ['battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc', 'four_g',
       'int_memory', 'm_dep', 'mobile_wt', 'n_cores', 'pc', 'px_height',
       'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time', 'three_g',
       'touch_screen', 'wifi']
data_modelling = dcopy_new.drop(useless_col, axis = 1)

y = data_modelling['price_range']
X1 = data_modelling.drop('price_range', axis = 1)
X2 = pd.get_dummies(data_modelling)
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1,y,random_state=42,test_size=0.43)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2,y,random_state=42,test_size=0.43)

# import the metrics class

algorithms = ['Random Forest', 'Decision Tree', 'Support Vector Machine']
metrics    = ['Confusion Matrix', 'Classification Report','Accuracy']
train_scores = {}
pd.set_option('display.max_rows', 10)

X=dcopy.drop(['price_range'],axis=1)
y=dcopy[['price_range']]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=101)


from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)
st.write("Decision Tree Score Is:")
st.write (dtree.score(X_test,y_test))


y = data_modelling['price_range']
X1 = data_modelling.drop('price_range', axis = 1)
X2 = pd.get_dummies(data_modelling)
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1,y,random_state=42,test_size=0.2)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2,y,random_state=42,test_size=0.2)

# import the metrics class

algorithms = ['Random Forest', 'Decision Tree', 'Support Vector Machine']
metrics    = ['Confusion Matrix', 'Classification Report','Accuracy']
train_scores = {}
pd.set_option('display.max_rows', 10)

def algorithm_validation(Algorithm=algorithms, Metrics=metrics):        
    if Algorithm == 'Random Forest':
        model = RandomForestClassifier(max_depth=2, random_state=0)
        model.fit(X_train2, y_train1) 
        y_pred = model.predict(X_test2)
        X_test1['Predict'] = model.predict(X_test2)
        
    elif Algorithm == 'Decision Tree':
        model = DecisionTreeClassifier(random_state=0)
        model.fit(X_train2, y_train1) 
        y_pred = model.predict(X_test2)
        X_test1['Predict'] = model.predict(X_test2)
    
    elif Algorithm == 'Support Vector Machine':
        model = SVC(kernel='linear')
        model.fit(X_train2, y_train1) 
        y_pred = model.predict(X_test2)
        X_test1['Predict'] = model.predict(X_test2)
        
    if Metrics == 'Classification Report':
        score = classification_report(y_test2, y_pred)
        
    elif Metrics == 'Accuracy':
        score = accuracy_score(y_test2, y_pred)
        
    elif Metrics == 'Confusion Matrix':
        plot_confusion_matrix(model, X_test2, y_test2)
        score = confusion_matrix(y_test2, y_pred)
        
    return print('\nThe ' + Metrics + ' of ' + Algorithm + ' is:\n\n'+ str(score) + '\n')


# Train the model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

# Logistic Regression

# Without reduction
logregwithoutpca = LogisticRegression()
logregwithoutpca.fit(X_train, y_train)

logregwithoutpca_result = logregwithoutpca.predict(X_test)#After training-need to perdict

st.write('Accuracy of Logistic Regression (without PCA) on training set: {:.2f}'
     .format(logregwithoutpca.score(X_train, y_train)))
st.write('Accuracy of Logistic Regression (without PCA)  on testing set: {:.2f}'
     .format(logregwithoutpca.score(X_test, y_test)))
st.write('\nConfusion matrix :\n',confusion_matrix(y_test, logregwithoutpca_result))
st.write('\n\nClassification report :\n\n', classification_report(y_test, logregwithoutpca_result))


st.write("Accuracy:Model Evaluation using Confusion Matrix")

from sklearn.model_selection import train_test_split
X=dcopy_new.drop('price_range',axis=1)
y=dcopy_new['price_range']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
y = data_modelling['price_range']
X1 = data_modelling.drop('price_range', axis = 1)
X2 = pd.get_dummies(data_modelling)
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1,y,random_state=42,test_size=0.2)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2,y,random_state=42,test_size=0.2)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



#ConfusionMatrix 

st.write("Visualization Confusion Matrix")
confusion_matrix =confusion_matrix(y_test, logregwithoutpca_result)
sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap='Set3')
plt.title('Confusion Matrix for KNN')
plt.xlabel('Predicted')
plt.ylabel('True')
st.write(classification_report(y_test, logregwithoutpca_result))

fig, ax = plt.subplots()
sns.heatmap(confusion_matrix, ax=ax)
st.pyplot(fig)


st.write("ConfusionMatrix In Percentage")
sns.heatmap(confusion_matrix/np.sum(confusion_matrix), annot=True, 
            fmt='.1%', cmap='Accent')
plt.title('Confusion Matrix for KNN In Percentage')
plt.xlabel('Predicted Value')
plt.ylabel('True')

fig, ax = plt.subplots()
sns.heatmap(confusion_matrix, ax=ax)
st.pyplot(fig)




group_counts = ["{0:0.0f}".format(value) for value in
                confusion_matrix.flatten()]
group_percentages = ["{0:.1%}".format(value) for value in
                     confusion_matrix.flatten()/np.sum(confusion_matrix)]

labels = [f"{v2}\n{v3}" for v2, v3 in
          zip(group_counts,group_percentages)]
labels = np.asarray(labels).reshape(4,4)
sns.heatmap(confusion_matrix, annot=labels, fmt='', cmap='Pastel1')

plt.title('Confusion Matrix for KNN ')

fig, ax = plt.subplots()
sns.heatmap(confusion_matrix, ax=ax)
st.pyplot(fig)




plt.clf()
plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Pastel2)

classNames = ['Negative','Positive','Positive','Positive']
plt.title('Mobile Phone Perdiction Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted ')
tick_marks = np.arange(len(classNames))
plt.xticks(tick_marks, classNames, rotation=45)
plt.yticks(tick_marks, classNames)
s = [['TN','FP','TP','TP'], ['TN','FP','TP','TP'],['TN','FP','TP','TP'],['TN','FP','TP','TP']]
for i in range(4):
    for j in range(4):
        plt.text(j,i, str(s[i][j])+" = "+str(confusion_matrix[i][j]))
plt.show()

fig, ax = plt.subplots()
sns.heatmap(confusion_matrix, ax=ax)
st.pyplot(fig)


