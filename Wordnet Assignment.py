#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing the necessary library.

from nltk.stem.snowball import SnowballStemmer
sn_stemmer=SnowballStemmer("english")
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import pandas as pd
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from sklearn.cluster import KMeans
import numpy as np

#wordcloud imports
from wordcloud import WordCloud
from wordcloud import ImageColorGenerator
from wordcloud import STOPWORDS
import matplotlib.pyplot as plt


# In[2]:


def datacleaning(x):
#---------------------------------------------------------------------------------------------------------------------------    
    #Removing stop words.
    y=nltk.word_tokenize(x)
    temp=[]
    for i in y:
      if i in stopwords.words("english"): 
         pass
      else:
         temp.append(i)
    my_new_string=' '.join(temp)
#---------------------------------------------------------------------------------------------------------------------------        
    #Lemmatizing the data.
    c=nltk.word_tokenize(my_new_string)
    my_list=[]
    for i in c:
        my_list.append(lemmatizer.lemmatize(i))
        my_new_string_second=' '.join(my_list)
#---------------------------------------------------------------------------------------------------------------------------            
#Stemming the data.
        d=nltk.word_tokenize(my_new_string_second)
    my_list_stemmer=[]
    for i in d:
        my_list_stemmer.append(sn_stemmer.stem(i))
        final=' '.join(my_list_stemmer)
#---------------------------------------------------------------------------------------------------------------------------  
    #Removing the numbers from the data.
        temp=[]
    for i in final:
        if i.isdigit(): 
            pass
        else:
            temp.append(i)
    hi=''.join(temp)
    return (hi.lower()) #Converting the data to lower case


# In[5]:


df = pd.read_csv('Restaurant_Reviews - Restaurant_Reviews.tsv',sep='\t')


# In[6]:


df['Data_ready']=df['Review'].apply(datacleaning)


# In[7]:


from sklearn.feature_extraction.text import TfidfVectorizer
hi_vectorize= TfidfVectorizer()


# In[8]:


hi_vectorize.fit(df['Data_ready'])


# In[9]:


hi_vectorize.get_feature_names_out()


# In[10]:


i_m_vector = hi_vectorize.transform(df['Data_ready'])


# In[11]:


i_m_vector


# In[12]:


i_m_vector.toarray()


# In[13]:


from sklearn.cluster import KMeans
import numpy as np


# In[14]:


# Initialising
km = KMeans(n_clusters=10)


# In[15]:


# Fitting a model
km_model = km.fit(i_m_vector)


# In[16]:


# Making predictions
y_km = km.predict(i_m_vector)


# In[17]:


y_km


# In[18]:


# calculate distortion for a range of number of cluster
import matplotlib.pyplot as plt
temp=[]
for i in range(9):
    #Initialising
    km=KMeans(n_clusters=i+1)
    #fitting the plot
    km_model=km.fit(i_m_vector)
    output=km_model.inertia_
    temp.append(output)


# In[19]:


import matplotlib.pyplot as plt

distortions = []
for i in range(1, 35):
    km = KMeans(
        n_clusters=i, init='random',
        n_init=1, max_iter=20,
    )
    km.fit(i_m_vector)
    distortions.append(km.inertia_)

# plot
plt.plot(range(1, 35), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.show()


# In[20]:


plt.plot(temp)


# In[21]:


from wordcloud import WordCloud
from wordcloud import ImageColorGenerator
from wordcloud import STOPWORDS
import matplotlib.pyplot as plt


# In[22]:


df.head()


# In[23]:


df


# In[24]:


df['Data_ready']


# In[25]:


df['Data_ready'].values


# In[26]:


type(df['Data_ready'].values)


# In[27]:


df.shape


# In[28]:


str(df['Data_ready'].values)


# In[29]:


len(str(df['Data_ready'].values))


# In[30]:


text_array = df['Data_ready'].values 


# In[31]:


text = str(text_array)


# In[32]:


text


# In[33]:


stopwords = set(STOPWORDS)


# In[34]:


stopwords


# In[35]:


WordCloud(background_color="white").generate(text)


# In[36]:


wordcloud = WordCloud(background_color="white").generate(text)


# In[37]:


wordcloud = WordCloud(stopwords=stopwords, background_color="pink").generate(str(text))


# In[38]:


plt.figure(figsize=(15,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[39]:


y_km


# In[40]:


df['Cluster']=y_km


# In[41]:


df_2 = df.groupby (['Cluster']).Data_ready.value_counts () 


# In[42]:


df_2


# In[43]:


my_list=[df_2]


# In[44]:


my_list


# In[45]:


cluster_0 = df[(df["Cluster"] == 0)]
cluster_1 = df[(df["Cluster"] == 1)]
cluster_2 = df[(df["Cluster"] == 2)]
cluster_3 = df[(df["Cluster"] == 3)]
cluster_4 = df[(df["Cluster"] == 4)]
cluster_5 = df[(df["Cluster"] == 5)]
cluster_6 = df[(df["Cluster"] == 6)]
cluster_7 = df[(df["Cluster"] == 7)]
cluster_8 = df[(df["Cluster"] == 8)]
cluster_9 = df[(df["Cluster"] == 9)]
cluster_10 = df[(df["Cluster"] == 10)]


# In[46]:


text_array_0 = cluster_0['Data_ready'].values 
text_array_1 = cluster_1['Data_ready'].values 
text_array_2 = cluster_2['Data_ready'].values 
text_array_3 = cluster_3['Data_ready'].values 
text_array_4 = cluster_4['Data_ready'].values 
text_array_5 = cluster_5['Data_ready'].values 
text_array_6 = cluster_6['Data_ready'].values 
text_array_7 = cluster_7['Data_ready'].values 
text_array_8 = cluster_8['Data_ready'].values 
text_array_9 = cluster_9['Data_ready'].values 
text_array_10 = cluster_10['Data_ready'].values 


# In[47]:


text_0 = str(text_array_0)
text_1 = str(text_array_1)
text_2 = str(text_array_2)
text_3 = str(text_array_3)
text_4 = str(text_array_4)
text_5 = str(text_array_5)
text_6 = str(text_array_6)
text_7 = str(text_array_7)
text_8 = str(text_array_8)
text_9 = str(text_array_9)
text_10 = str(text_array_10)


# In[48]:


wordcloud_0 = WordCloud(stopwords=stopwords, background_color="purple").generate(str(text_0))
wordcloud_1 = WordCloud(stopwords=stopwords, background_color="indigo").generate(str(text_1))
wordcloud_2= WordCloud(stopwords=stopwords, background_color="green").generate(str(text_2))
wordcloud_3= WordCloud(stopwords=stopwords, background_color="orange").generate(str(text_3))
wordcloud_4= WordCloud(stopwords=stopwords, background_color="red").generate(str(text_4))
wordcloud_5= WordCloud(stopwords=stopwords, background_color="blue").generate(str(text_5))
wordcloud_6= WordCloud(stopwords=stopwords, background_color="grey").generate(str(text_6))
wordcloud_7= WordCloud(stopwords=stopwords, background_color="brown").generate(str(text_7))
wordcloud_8= WordCloud(stopwords=stopwords, background_color="white").generate(str(text_8))
wordcloud_9= WordCloud(stopwords=stopwords, background_color="black").generate(str(text_9))


# In[49]:


plt.imshow(wordcloud_0, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[50]:


plt.imshow(wordcloud_1, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[51]:


plt.imshow(wordcloud_2, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[52]:


plt.imshow(wordcloud_3, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[53]:


plt.imshow(wordcloud_4, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[54]:


plt.imshow(wordcloud_5, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[55]:


plt.imshow(wordcloud_6, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[56]:


plt.imshow(wordcloud_7, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[57]:


plt.imshow(wordcloud_8, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[58]:


plt.imshow(wordcloud_9, interpolation='bilinear')
plt.axis("off")
plt.show()

