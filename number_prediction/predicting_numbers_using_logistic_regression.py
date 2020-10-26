#!/usr/bin/env python
# coding: utf-8

# In[77]:


from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
get_ipython().run_line_magic('matplotlib', 'inline')
digits = load_digits()


# In[78]:


print("Image Data Shape",digits.data.shape)
print("Label Data Shape",digits.target.shape)


# In[79]:


plt.figure(figsize=(20,4))
for index,(image, label) in enumerate(zip(digits.data[0:10], digits.target[0:10])):
    plt.subplot(1,10,index+1)
    plt.imshow(np.reshape(image,(8,8)),cmap=plt.cm.gray)
    plt.title('Training: %i\n' % label, fontsize=20)


# In[81]:


X_train,X_test,Y_train,Y_test = train_test_split(digits.data,digits.target,test_size=0.2,random_state=2)


# In[82]:


print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)


# In[83]:


from sklearn.linear_model import LogisticRegression
regressor=LogisticRegression()
regressor.fit(X_train,Y_train)


# In[84]:


print(regressor.predict(X_test))


# In[85]:


predictions = regressor.predict(X_test)


# In[86]:


score = regressor.score(X_test,Y_test)
print(score)


# In[87]:


cm = metrics.confusion_matrix(Y_test, predictions)
print(cm)


# In[88]:


plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidth=.5,square=True, cmap="Blues_r")
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title,size=15)


# In[89]:


index = 0
missClassifiedIndex = []
for predict,actual in zip(predictions,Y_test):
    if predict==actual:
        missClassifiedIndex.append(index)
    index+=1
plt.figure(figsize=(20,3))
for plotIndex,wrong in enumerate(missClassifiedIndex[0:5]):
    plt.subplot(1,5,plotIndex+1)
    plt.imshow(np.reshape(X_test[wrong],(8,8)),cmap=plt.cm.gray)
    plt.title("Predicted: {},Actual: {}".format(predictions[wrong],Y_test[wrong]))

