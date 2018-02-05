#import libraries
import numpy as np
import pandas as pd

#import dataset
dataset = pd.read_csv('Data.csv')

#Spliting the dependent variables and independent variables
Ind_var = dataset.iloc[:,9:13].values
Dep_var = dataset.iloc[:,-1].values

#for demo of cat var just extracting these variable 
'''
These independent variable are encoded only when theyhave some effect in
the precited value. for example if the parent satisfaction effect the final
outcome then we shall use it
'''
cat_demo_var = dataset.iloc[:,0].values


# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(Ind_var[:, 0:5])
Ind_var[:, 0:5] = imputer.transform(Ind_var[:, 0:5])


# Encoding categorical data'
'''For this Dataset for independent variables there is no cat var, so lets take
other variable for instance'''
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
cat_demo_var[:,0] = labelencoder_X.fit_transform(cat_demo_var[:,0])
onehotencoder = OneHotEncoder(categorical_features = [0])
en_cat = onehotencoder.fit_transform(cat_demo_var).toarray()


# Encoding the dependent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_Dep = LabelEncoder()
Dep_var = labelencoder_Dep.fit_transform(Dep_var)

#Feautre scaling
from sklearn.preprocessing import StandardScaler
scl = StandardScaler()
Ind_var_scale = scl.fit_transform(Ind_var)  

# Sampling the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
ind_train, ind_test, dep_train, dep_test = train_test_split(Ind_var, Dep_var, test_size = 0.2, random_state = 0)


