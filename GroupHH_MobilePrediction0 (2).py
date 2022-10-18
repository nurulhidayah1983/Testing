import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.header("Mobile Prediction project")

data = pd.read_csv('/content/train.csv',)
st.write(data.head())


st.write (data.isna().sum())

st.write (data.columns)



fig, axs = plt.subplots(22, figsize = (10,25))
colors = ['#0000FF', '#00FF00',
          '#FFFF00', '#FF00FF']


st.pyplot(plt1 = sns.boxplot(data['battery_power'], ax = axs[0]))
st.pyplot(plt2 = sns.boxplot(data['blue'], ax = axs[1]))
st.pyplot(plt3 = sns.boxplot(data['clock_speed'], ax = axs[2]))
st.pyplot(plt4 = sns.boxplot(data['dual_sim'], ax = axs[3]))
st.pyplot(plt5 = sns.boxplot(data['fc'], ax = axs[4]))
st.pyplot(plt6 = sns.boxplot(data['four_g'], ax = axs[5]))
st.pyplot(plt7 = sns.boxplot(data['int_memory'], ax = axs[6]))
st.pyplot(plt8 = sns.boxplot(data['m_dep'], ax = axs[7]))
st.pyplot(plt9 = sns.boxplot(data['mobile_wt'], ax = axs[8]))
st.pyplot(plt10 = sns.boxplot(data['n_cores'], ax = axs[9]))
st.pyplot(plt11 = sns.boxplot(data['pc'], ax = axs[10]))
st.pyplot(plt12 = sns.boxplot(data['px_height'], ax = axs[11]))
st.pyplot(plt13 = sns.boxplot(data['px_width'], ax = axs[12]))
st.pyplot(plt14 = sns.boxplot(data['ram'], ax = axs[13]))
st.pyplot(plt15 = sns.boxplot(data['four_g'], ax = axs[14]))
st.pyplot(plt16 = sns.boxplot(data['int_memory'], ax = axs[15]))
st.pyplot(plt17 = sns.boxplot(data['sc_h'], ax = axs[16]))
st.pyplot(plt18 = sns.boxplot(data['talk_time'], ax = axs[17]))
st.pyplot(plt19= sns.boxplot(data['three_g'], ax = axs[18]))
st.pyplot(plt20 = sns.boxplot(data['touch_screen'], ax = axs[19]))
st.pyplot(plt21 = sns.boxplot(data['wifi'], ax = axs[20]))
st.pyplot(plt22 = sns.boxplot(data['price_range'], ax = axs[21]))

st.pyplot(plt.tight_layout())



st.write(data.info())

import numpy as np

data1=data["fc"].mean()
data["fc"] =np.where(data["fc"] >16,mean,data['fc'])
st.write(print('Mean of the fc dataset is replacing outlier the value as below:'))
st.write(print(data1))

data3=data["px_height"].mean()
data["px_height"] =np.where(data["px_height"] >1750,mean,data['px_height'])
st.write(print('Mean of the px_height dataset is replacing outlier the value as below:'))
st.write(print(data3))


st.write(data.describe())


fig, axs = plt.subplots(2, figsize = (5,7))
colors = ['#0000FF', '#00FF00',
          '#FFFF00', '#FF00FF']

st.pyplot(plt5 = sns.boxplot(data['fc'], ax = axs[0]))
st.pyplot(plt5 = sns.boxplot(data['px_height'], ax = axs[1]))

st.write(plt.tight_layout())




data.duplicated().any()



st.write(data.info())




from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score


st.write(data.info())
dcopy=data.copy()

 
st.write(dcopy=data.copy())



st.write(dcopy)

st.write(dcopy.shape)

st.write(dcopy.columns)

st.write(dcopy.head())

st.write(dcopy.dtypes)


dcopy_catagorical=dcopy.select_dtypes(include=['object']).columns.tolist()
dcopy_catagorical


dcopy.describe()



dcopy_new=dcopy

dcopy_new[['clock_speed', 'm_dep']] = dcopy[['clock_speed', 'm_dep']].astype('int64') #converting data type into int64

st.write(dcopy_new.dtypes)


dcopy_new.shape
st.write(data.shape)

dcopy.to_csv("Train_ProjectVersion.csv")



st.write(dcopy.corr()['price_range'])

matrix = dcopy.corr()
f, ax = plt.subplots(figsize=(20, 15))
st.pyplot(sns.heatmap(matrix, vmax=1, square=True, annot=True))



st.pyplot(sns.distplot(data['price_range']);)

st.write(data.describe())

st.write(data.info())

st.write(data.shape)

st.write(data.columns)




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.core.pylabtools import figsize

import folium
from folium.plugins import HeatMap
import plotly.express as px
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score

def transform(dataframe):      
      
    le = LabelEncoder()
    categorical_features = list(dataframe.columns[dataframe.dtypes ==np.int64 ])    
    return dataframe[categorical_features].apply(lambda x: le.fit_transform(x))

X = dcopy_new.drop('ram', axis=1)
Xin=transform(X)
y = dcopy_new['price_range']
X_train, X_test, y_train, y_test = train_test_split(Xin, y, test_size=0.2, random_state=42)#change test size to 20%.
ss = StandardScaler()
X_train_scaled = ss.fit_transform(X_train)
X_test_scaled = ss.transform(X_test)
pca = PCA().fit(X_train_scaled)

st.write(loadings = pd.DataFrame(
    data=pca.components_.T * np.sqrt(pca.explained_variance_), 
    columns=[f'PC{i}' for i in range(1, len(X_train.columns) + 1)],
    index=X_train.columns
)
loadings)


pc1_loadings = loadings.sort_values(by='PC1', ascending=False)[['PC1']]
pc1_loadings = pc1_loadings.reset_index()
pc1_loadings.columns = ['Attribute', 'CorrelationWithPC1']

st.pyplot(plt.bar(x=pc1_loadings['Attribute'], height=pc1_loadings['CorrelationWithPC1'], color='#087E8B'))plt.title('PCA loading scores (first principal component)', size=20)
st.pyplot(plt.xticks(rotation='vertical'))
st.pyplot(plt.show())



import seaborn as seaborns

st.pyplot(sns.countplot(dcopy_new['price_range']))

st.write(dcopy_new.columns)


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


algorithms = ['Random Forest', 'Decision Tree', 'Support Vector Machine']
metrics    = ['Confusion Matrix', 'Classification Report','Accuracy']
algorithm_validation('Random Forest','Classification Report')


st.write(algorithm_validation('Decision Tree','Classification Report'))


st.write(algorithm_validation('Support Vector Machine','Classification Report'))

from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)
st.write(dtree.score(X_test,y_test))


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


from sklearn.decomposition import PCA

pca = PCA().fit(X_train)
st.pyplot((plt.plot(np.cumsum(pca.explained_variance_ratio_)))
st.pyplot((plt.xlabel('number of components'))
st.pyplot((plt.ylabel('cumulative explained variance');)

accum_explained_var = np.cumsum(pca.explained_variance_ratio_)

min_threshold = np.argmax(accum_explained_var > 0.90)

st.write(min_threshold)


pca = PCA(n_components = min_threshold + 1)

X_train_projected= pca.fit_transform(X_train)
X_test_projected = pca.transform(X_test)

st.write(X_train_projected.shape)


from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix, classification_report


logregwithoutpca = LogisticRegression()
logregwithoutpca.fit(X_train, y_train)

logregwithoutpca_result = logregwithoutpca.predict(X_test)#After training-need to perdict

st.write(print('Accuracy of Logistic Regression (without PCA) on training set: {:.2f}'
     .format(logregwithoutpca.score(X_train, y_train))))
st.write(print('Accuracy of Logistic Regression (without PCA)  on testing set: {:.2f}'
     .format(logregwithoutpca.score(X_test, y_test))))
st.write(print('\nConfusion matrix :\n',confusion_matrix(y_test, logregwithoutpca_result)))
st.write(print('\n\nClassification report :\n\n', classification_report(y_test, logregwithoutpca_result)))





