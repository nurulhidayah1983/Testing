

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('train.csv',)
data.head()

data.isna().sum()
data.columns

fig, axs = plt.subplots(22, figsize = (10,25))
colors = ['#0000FF', '#00FF00',
          '#FFFF00', '#FF00FF']


plt1 = sns.boxplot(data['battery_power'], ax = axs[0])
plt2 = sns.boxplot(data['blue'], ax = axs[1])
plt3 = sns.boxplot(data['clock_speed'], ax = axs[2])
plt4 = sns.boxplot(data['dual_sim'], ax = axs[3])
plt5 = sns.boxplot(data['fc'], ax = axs[4])
plt6 = sns.boxplot(data['four_g'], ax = axs[5])
plt7 = sns.boxplot(data['int_memory'], ax = axs[6])
plt8 = sns.boxplot(data['m_dep'], ax = axs[7])
plt9 = sns.boxplot(data['mobile_wt'], ax = axs[8])
plt10 = sns.boxplot(data['n_cores'], ax = axs[9])
plt11 = sns.boxplot(data['pc'], ax = axs[10])
plt12 = sns.boxplot(data['px_height'], ax = axs[11])
plt13 = sns.boxplot(data['px_width'], ax = axs[12])
plt14 = sns.boxplot(data['ram'], ax = axs[13])
plt15 = sns.boxplot(data['four_g'], ax = axs[14])
plt16 = sns.boxplot(data['int_memory'], ax = axs[15])
plt17 = sns.boxplot(data['sc_h'], ax = axs[16])
plt18 = sns.boxplot(data['talk_time'], ax = axs[17])
plt19= sns.boxplot(data['three_g'], ax = axs[18])
plt20 = sns.boxplot(data['touch_screen'], ax = axs[19])
plt21 = sns.boxplot(data['wifi'], ax = axs[20])
plt22 = sns.boxplot(data['price_range'], ax = axs[21])

plt.tight_layout()

data.info()

data["fc"] = np.where(data["fc"] >16,mean,data['fc'])
data["px_height"] = np.where(data["px_height"] >1750,mean,data['px_height'])
data.describe()

fig, axs = plt.subplots(2, figsize = (5,7))
colors = ['#0000FF', '#00FF00',
          '#FFFF00', '#FF00FF']

plt5 = sns.boxplot(data['fc'], ax = axs[0])
plt5 = sns.boxplot(data['px_height'], ax = axs[1])

plt.tight_layout()


data.duplicated().any()

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score

data.info()
dcopy=data.copy()
dcopy.shape
dcopy.columns
dcopy.describe()

dcopy_new=dcopy
dcopy_new[['clock_speed', 'm_dep']] = dcopy[['clock_speed', 'm_dep']].astype('int64')
dcopy_new.dtypes

dcopy_new.shape
data.shape

dcopy.corr()['price_range']
matrix = dcopy.corr()
f, ax = plt.subplots(figsize=(20, 15))
sns.heatmap(matrix, vmax=1, square=True, annot=True)

sns.distplot(data['price_range']);

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

loadings = pd.DataFrame(
    data=pca.components_.T * np.sqrt(pca.explained_variance_), 
    columns=[f'PC{i}' for i in range(1, len(X_train.columns) + 1)],
    index=X_train.columns
)
loadings


pc1_loadings = loadings.sort_values(by='PC1', ascending=False)[['PC1']]
pc1_loadings = pc1_loadings.reset_index()
pc1_loadings.columns = ['Attribute', 'CorrelationWithPC1']

plt.bar(x=pc1_loadings['Attribute'], height=pc1_loadings['CorrelationWithPC1'], color='#087E8B')
plt.title('PCA loading scores (first principal component)', size=20)
plt.xticks(rotation='vertical')
plt.show()

import seaborn as seaborns
sns.countplot(dcopy_new['price_range'])

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

algorithm_validation('Decision Tree','Classification Report')
algorithm_validation('Support Vector Machine','Classification Report')

from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)
dtree.score(X_test,y_test)


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
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');

accum_explained_var = np.cumsum(pca.explained_variance_ratio_)

min_threshold = np.argmax(accum_explained_var > 0.90) # use 90%-cannot be higher,unable to display

min_threshold
pca = PCA(n_components = min_threshold + 1)

X_train_projected= pca.fit_transform(X_train)
X_test_projected = pca.transform(X_test)

X_train_projected.shape
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
logregwithoutpca = LogisticRegression()
logregwithoutpca.fit(X_train, y_train)

logregwithoutpca_result = logregwithoutpca.predict(X_test)#After training-need to perdict

print('Accuracy of Logistic Regression (without PCA) on training set: {:.2f}'
     .format(logregwithoutpca.score(X_train, y_train)))
print('Accuracy of Logistic Regression (without PCA)  on testing set: {:.2f}'
     .format(logregwithoutpca.score(X_test, y_test)))
print('\nConfusion matrix :\n',confusion_matrix(y_test, logregwithoutpca_result))
print('\n\nClassification report :\n\n', classification_report(y_test, logregwithoutpca_result))

sns.jointplot(x='ram',y='price_range',data=dcopy_new,color='red',kind='kde');
sns.pointplot(y="int_memory", x="price_range", data=dcopy_new)