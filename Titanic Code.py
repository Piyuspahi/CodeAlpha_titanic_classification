
'''CRISP-ML(Q):
    1.a.i. Business problem: Predict The survival rate of titanic ship pssengers
        ii. Business Objectives: Identifay the survival rate
        iii. Business Constraints: Reduce the inaccurate predaction.
        Success Criteria:
        i. Business success criteria: Reduce the inaccurate predaction.
        ii. ML success criteria: Achieve an accuracy of over 95%
        iii. Economic success criteria:
             
    1.b. Data Collection: Bank -> 418 rows, 12 column (11 Inputs and 1 Ouput)
    2. Data Preprocessing - Cleansing & EDA / Descriptive Analytics
    3. Model Building - Experiment with different models alongside Hyperparameter tuning
    4. Evaluation - Not just model evaluation based on accuracy but we also need 
       to evaluate business & economic success criteria
    5. Model Deployment (Flask)
    6. Monitoring & Maintenance (Prediction results to the database - MySQL / MS SQL)'''



import pandas as pd # deals with data frame        # for Data Manipulation"
import numpy as np  # deals with numerical values  # for Mathematical calculations"
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sqlalchemy import create_engine
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from feature_engine.outliers import Winsorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import statsmodels.formula.api as smf
from sklearn.preprocessing import PolynomialFeatures



data = pd.read_csv(r"C:\Users\Piyus Pahi\Documents\Code Alpha Data Science Project\Tatinic Data\tested.csv")
print(data.head())

engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}"
                       .format(user="root",# user
                               pw="965877", # passwrd
                               db="titanic_db")) #database


data.to_sql('titanic',con = engine, if_exists = 'replace', index = False)

# EDA
data.info()

data.describe()

data.columns

data.head(10)

data.tail(10)

##checking duplicate values
#data.duplicated().sum()
# Checking for Null values
data.isnull().sum()

##remove null values
df = data.dropna()
df.isnull().sum()



# ### AutoEDA
##############
# sweetviz
##########
import sweetviz
my_report = sweetviz.analyze([data, "data"])
my_report.show_html('Report1.html')

# D-Tale
########
import dtale
d = dtale.show(data)
d.open_browser()

##drop some unrequired columns
data = data.drop(columns = ['Name','Ticket','Fare','Cabin','Embarked'])
data

## plot graph for survier
pclass_sur = data[['Pclass','Survived']].groupby('Pclass').sum()
pclass_sur

pclass_sur.plot(kind = 'bar')
plt.title('survivers per class')
plt.ylabel('survived number')

sex_sur = data[['Sex', 'Survived']].groupby('Sex').sum()
sex_sur

sex_sur.plot(kind = 'bar')
plt.title('Sex survived per class')
plt.ylabel('survived number')

#Split the data set in inpt and output
x = data.drop(columns = 'Survived')
x

y = data['Survived']
y

##apply labelencoding
label = LabelEncoder()
x['Sex']=label.fit_transform( x['Sex'])

##split data in to tran and test data
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 42)

##load random forest model
model = RandomForestClassifier(n_estimators = 100,random_state = 42)
model.fit(x_train,y_train)

##predict the value
y_pred = model.predict(x_test)
y_pred

##evaluate the accuracy score
print('Accuracy',accuracy_score(y_test,y_pred))



































































































































































































































































































