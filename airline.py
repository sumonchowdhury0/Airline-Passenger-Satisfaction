import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import seaborn as sns
from pandas.api.types import is_string_dtype

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

from sklearn import tree
from sklearn.tree import export_graphviz
from sklearn.linear_model import LogisticRegression
import statsmodels
import statsmodels.api as sm
import pydotplus
from IPython.display import Image  
import graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
import os

# In[1]

os.chdir(r"C:\Users\schow\Desktop\Imarticus\Ml\Ml deployment")

df= pd.read_csv("airline_passenger_satisfaction.csv")
df=pd.DataFrame(df)
df.head()

# In[2]

df.info()

# In[3]

X = df[["age","ease_of_online_booking","gate_location","food_and_drink","online_boarding","seat_comfort",
        "baggage_handling","inflight_service","cleanliness"]]

X.head()
# In[4]

y=df.satisfaction
y.head()
# In[5]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=1)

# In[6]
model = SGDClassifier()
model.fit(X_train,y_train)

# In[7]
from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# In[8]
import pickle
pickle.dump(model,open("model.pkl","wb"))

model = pickle.load(open("model.pkl","rb"))
print(model.predict([[26,2,2,5,5,5,4,4,5]]))

#26,2,2,5,5,5,4,4,5 ---  satisfied
#46,2,2,1,2,1,2,3,2---   dissatisfied




