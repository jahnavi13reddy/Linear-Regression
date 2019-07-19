# Import the libraries

#In[1]

#scipy used for scientific calculation 
import pandas as pd
import numpy as np #deals numbers and calculation scipy is also the same 
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline  
#brings backend activities to front end 


# Set Warnings

#In[2]:

import warnings
warnings.filterwarnings('ignore')


#Set local path to access data set

#In [3]:

#the path where the USAhousing document has been saved in your system
cd F:/Downloads/machine learning

# when we run this particular line the directory changes to the following recpective to the path include by you
F:\Downloads\machine learning

#Load the data set

#In[4]:

USAhousing = pd.read_csv('USA_Housing.csv')


#Interpreting the data

#In[5]:

# by default it displays first 5 Rows. You can also give head(30) etc.., any value within the range of number of rows
USAhousing.head()
#see screenshot scrn1 for sample output

#In[6]:

# by default it displays last 5 Rows. You can also give tail(30) etc.., any value within the range of number of rows
USAhousing.tail()
#see screenshot scrn2 for sample output

#In[7]:

USAhousing.info()
#see screenshots scrn3 for sample output

#In[8]:

USAhousing.describe()
#see screenshots scrn4 for  output

#In[9]:

USAhousing.columns
#see screenshots scrn5 for output


#EDA(exploratory data analysis)
#Let's create some simple plots to check out the data

#In[10]:
#see screenshots scrn6 for sample output
sns.pairplot(USAhousing)

#In[11]:
#see screenshots scrn7 for sample output

#In[12]:
#see screenshots scrn8 for sample output
sns.distplot(USAhousing['Price'])

#In[13]:
#see screenshots for scrn9 sample output
sns.heatmap(USAhousing.corr(),annot=True)

#splitting X and Y arrays
#In[14]:

x = USAhousing[['Avg. Area Income',
                'Avg. Area House Age',
                'Avg. Area Number of Rooms',
                'Avg. Area Number of Bedrooms',
                'Area Population']]
y = USAhousing['Price']


#Train Test Split
#now lets split the data into a training set. we will train our model on the training set 

#In[15]:

from sklearn.model_selection import train_test_split

#In[16]:

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)


#Algorithm Selection

#In[17]:

from sklearn.linear_model import LinearRegression
#for selecting the model


#Creating and Training the model
#In[18]:

lm = LinearRegression()
# creating object for model lm initiated

#In[19]:

lm.fit(x_train,y_train)
#fixing my model

#Out[19]:
#LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None,
#         normalize=False)

# Model Evaluation

#In[20]:

# print the intercept
print(lm.intercept_)

#out[20]:
#-2638142.110430972

#In[21]:

mycoef = pd.DataFrame(lm.coef_,x.columns,columns=['Coefficient'])
mycoef
#see screenshots scrn9 for output sample

#Predictions from our Model

#In[22]:

predictions = lm.predict(x_test)
#In[23]:

plt.scatter(y_test,predictions)
# prediction based above cell execution
#see screenshots scrn10 for output sample
