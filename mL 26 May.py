import numpy as np
import pandas as pd

df=pd.read_csv("C:\\Users\\PARUL\\Downloads\\covid_toy - covid_toy.csv")

df.head()

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler ,OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


X_train,X_test,y_train,y_test = train_test_split(
df.drop('has_covid',axis=1),df['has_covid'],test_size=0.2,random_state=42)


#define the column that need to be preprocessed
categorical_features=['gender','city']
numeric_features=['age','fever']

#create transformer
categorical_transformer=Pipeline(steps=[
    ('imputer',SimpleImputer(strategy='most_frequent')),
    ('onehot',OneHotEncoder(handle_unknown='ignore'))
])

numeric_transformer=Pipeline(steps=[
    ('imputer',SimpleImputer(strategy='mean')),
    ('scaler',StandardScaler())
])


#cobination transformation
preprocessor= ColumnTransformer(
 transformers=[
     ('num',numeric_transformer,numeric_features),
     ('cat',categorical_transformer,categorical_features)
 ])
#create the pipeline 
clf=Pipeline(steps=[('preprocessor',preprocessor),
                   ('classifier',LogisticRegression())])
#Train the model
clf.fit(X_train,y_train)
#evalute the model
y_pred=clf.predict(X_test)


y_pred

from sklearn.metrics import accuracy_score

acc= accuracy_score(y_test,y_pred)

acc

# Complet_case_analysis

import numpy as np
import pandas as pd

df=pd.read_csv("C:\\Users\\PARUL\\Downloads\\dsjob - dsjob.csv")

df.head()

df.isnull().mean()*100

cols=[var for var in df.columns if df[var].isnull().mean()<0.05 and df[var].isnull().mean()>0]
cols

df[cols].sample(5)

df['education_level'].value_counts()

len(df[cols].dropna())/len(df)

new_df=df[cols].dropna()
df.shape ,new_df.shape

import matplotlib .pyplot as plt

fig =plt.figure()
ax=fig.add_subplot(111)

#original data
df['experience'].hist(bins=50,ax=ax, density =True ,color ='red')

# data after cca . the argunment alpha makes the coloor  transparent , so we can
# see the overlay of the 2 distribution
new_df['experience'].hist(bins=50,ax=ax, density =True ,color ='green',alpha=0.8)

temp= pd.concat([
    #percentage of observations per catagorey ,original data
    df['enrolled_university'].value_counts()/len(df),
    
    ##percentage of observations per catagorey ,cca data
    new_df['enrolled_university'].value_counts()/len(new_df),
    
]
,axis=1)


#add column names
temp.columns=['original','cca']
temp

