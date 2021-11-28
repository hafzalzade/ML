#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pyspark.sql import SparkSession
spark = SparkSession .builder .appName("test") .getOrCreate()


# In[2]:


df = spark.read.csv("/usr/local/Cellar/apache-spark/3.2.0/libexec/examples/src/main/resources/people.csv", header=True, sep=';')


# In[3]:


df.show()


# In[4]:


df.count()


# In[5]:


df.printSchema()


# In[6]:


df.select("name").show()


# In[7]:


spark = SparkSession.builder.appName("CTR").getOrCreate()


# In[8]:


from pyspark.sql.types import StructField, StringType, StructType, IntegerType


# In[9]:


schema=StructType([
StructField("id",StringType(),True),
StructField("click",IntegerType(),True),
StructField("hour",IntegerType(),True),
StructField("C1",StringType(),True),
StructField("banner_pos",StringType(),True),
StructField("site_id",StringType(),True),
StructField("site_domain",StringType(),True),
StructField("site_category",StringType(),True),
StructField("app_id",StringType(),True),
StructField("app_domain",StringType(),True),
StructField("app_category",StringType(),True),
StructField("device_id",StringType(),True),
StructField("device_ip",StringType(),True),
StructField("device_model",StringType(),True),
StructField("device_type",StringType(),True),
StructField("device_conn_type",StringType(),True),
StructField("C14",StringType(),True),
StructField("C15",StringType(),True),
StructField("C16",StringType(),True),
StructField("C17",StringType(),True),
StructField("C18",StringType(),True),
StructField("C19",StringType(),True),
StructField("C20",StringType(),True),
StructField("C21",StringType(),True),])


# In[10]:


df = spark.read.csv("file:/Users/hamidafzal/Documents/شخصی/etucation/arshad ut/1/machin_learning/homework6/train.csv", schema=schema,header=True)


# In[11]:


df.printSchema()


# In[12]:


df.count()


# In[13]:


df =df.drop('id').drop('hour').drop('device_id').drop('device_ip')


# In[14]:


df = df.withColumnRenamed("click", "label")


# In[15]:


df.columns


# In[16]:


df_train, df_test = df.randomSplit([0.7, 0.3], 42)


# In[17]:


df_train.cache()


# In[18]:


df_train.count()


# In[19]:


categorical = df_train.columns
categorical.remove('label')
print(categorical)


# In[20]:


from pyspark.ml.feature import StringIndexer
indexers = [
    StringIndexer(inputCol=c, outputCol=
                  "{0}_indexed".format(c)).setHandleInvalid("keep")
    for c in categorical]


# In[21]:


from pyspark.ml.feature import OneHotEncoder
encoder = OneHotEncoder(inputCols=[indexer.getOutputCol() for indexer in indexers], outputCols=["{0}_encoded".format(indexer.getOutputCol()) for indexer in indexers])


# In[22]:


from pyspark.ml.feature import VectorAssembler
assembler = VectorAssembler( inputCols=encoder.getOutputCols(), outputCol="features")


# In[23]:


stages = indexers + [encoder, assembler]
from pyspark.ml import Pipeline
pipeline = Pipeline(stages=stages)


# In[24]:


one_hot_encoder = pipeline.fit(df_train)
df_train_encoded = one_hot_encoder.transform(df_train)
df_train_encoded.show()


# In[25]:


df_train_encoded = df_train_encoded.select(["label", "features"])
df_train_encoded.show()


# In[26]:


df_train_encoded.cache()


# In[27]:


df_train.unpersist()


# In[28]:


df_test_encoded = one_hot_encoder.transform(df_test)
df_test_encoded = df_test_encoded.select(["label", "features"])
df_test_encoded.show()


# In[29]:


df_test_encoded.cache()
df_test.unpersist()


# In[30]:


from pyspark.ml.classification import LogisticRegression
classifier = LogisticRegression(maxIter=20, regParam=0.001,elasticNetParam=0.001)


# In[31]:


lr_model = classifier.fit(df_train_encoded)


# In[ ]:


predictions = lr_model.transform(df_test_encoded)


# In[ ]:


predictions.cache()
 predictions.show()

