from joblib.parallel import TASK_DONE
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import metrics
import seaborn as sns
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import joblib
from sklearn.preprocessing import StandardScaler
r=pd.read_csv("ad.csv")
print(r)
r.head()
r.tail()
r.shape
r.info()
r.isnull().sum()
r.describe()
r.head(1)
r.columns
x = r.drop('Chance of Admit ',axis=1)
y=r['Chance of Admit ']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=42)
y_train
r.head()
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)
x_train
r.head()
lr = LogisticRegression()
threshold = 0.5
y_train_binary = (y_train > threshold).astype(int)
y_test_binary = (y_test > threshold).astype(int)
lr = LogisticRegression()
lr.fit(x_train, y_train_binary)
y_pred1 = lr.predict(x_test)
svm = SVC(kernel='linear')
svm.fit(x_train, y_train_binary)
y_pred2 = svm.predict(x_test)
knn=KNeighborsClassifier()
knn.fit(x_train,y_train_binary)
y_pred3 = knn.predict(x_test)
rf = RandomForestClassifier()
rf.fit(x_train,y_train_binary)
y_pred4 = rf.predict(x_test)
print(accuracy_score(y_test_binary,y_pred4))
gr = GradientBoostingClassifier()
gr.fit(x_train,y_train_binary)
y_pred5 = gr.predict(x_test)
final_data = pd.DataFrame({'Models':['LR','SVC','KNN','RF','GBC'],
                          'ACC_SCORE':[accuracy_score(y_test_binary,y_pred1),
                                       accuracy_score(y_test_binary,y_pred2),
                                       accuracy_score(y_test_binary,y_pred3),
                                       accuracy_score(y_test_binary,y_pred4),
                                       accuracy_score(y_test_binary,y_pred5)]})
import seaborn as sns
sns.barplot(x=final_data['Models'],y=final_data['ACC_SCORE'])
r.columns
x = r.drop('Chance of Admit ',axis=1)
y = r['Chance of Admit ']
y = [1 if value>0.8 else 0 for value in y]
y = np.array(y)
gr = GradientBoostingClassifier()
gr.fit(x_train, y_train_binary)
y_pred = gr.predict(x_test)
accuracy = accuracy_score(y_test_binary, y_pred)
import joblib
joblib.dump(gr,'admission_model')
model = joblib.load('admission_model')
r.columns
model.predict(sc.transform([[337,118,4,4.5,4.5,9.65,1]]))
import streamlit as st

st.title('Graduate Admission Prediction App')

st.write('This app predicts the admission of graduate.')

st.header('Enter the features for prediction:')
with st.form(key='my_form'):
    X1 = st.number_input('Enter your GRE Score')
    X2 = st.number_input('Enter your TOEFL Score')
    X3 = st.number_input('Enter your University Rating')
    X4 = st.number_input('Enter your SOP')
    X5 = st.number_input('Enter your LOR')
    X6 = st.number_input('Enter your CGPA')
    X7 = st.number_input('Research')
    submit = st.form_submit_button(label='Submit')
if submit:
   s1=[X1,X2,X3,X4,X5,X6,X7]
   model = joblib.load('admission_model')
   result = model.predict(sc.transform([[X1,X2,X3,X4,X5,X6,X7]]))
   if result==1:
      st.write("High Chance of getting admission")
   else:
      st.write("You may get admission")