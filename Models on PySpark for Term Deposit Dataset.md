

```python
#import data file 

from pyspark.sql import SQLContext
sqlContext = SQLContext(sc)
spark_df = sqlContext.sql("Select * from TermDeposit_csv")
```


```python
#count the missing values
null_count_list = []
for i in spark_df.columns:
null_count = spark_df.select(i).where(col(i).isNull()).count()
null_count_list.append([i,null_count])
null_count_list
```


```python
#analyze churn distribution

import pandas as pd
#pandas_df = 
spark_df.select('deposit').groupBy('deposit').count().show()
#pandas_df.transpose()
```


```python
# summarize all variables 

summary = spark_df.describe().toPandas().transpose()
summary
```


```python
# define target variable and segregate categorical and numerical variables

target = 'deposit'
dtypes = spark_df.dtypes
cat_input = []
for i in range(0, len(spark_df.columns)):
if dtypes[i][1] == 'string':
    cat_input.append(dtypes[i][0])
cat_input = list(set(cat_input)-set(target))
cat_input

num_input = list(set(spark_df.columns) - set([target]) - set(cat_input))
num_input
```


```python
#Plotting the  correlations
import matplotlib.pyplot as plt
numeric_data = spark_df.select(num_input).toPandas()
axs = pd.scatter_matrix(numeric_data, figsize=(8, 8));
n = len(numeric_data.columns)
for i in range(n):
    v = axs[i, 0]
    v.yaxis.label.set_rotation(45)
    v.yaxis.label.set_ha('right')
    v.set_yticks(())
    h = axs[n-1, i]
    h.xaxis.label.set_rotation(45)
    h.set_xticks(())
display(plt.show())
```


```python
# Vizualize boxplot for numerical variables

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt1

numeric_data1 = spark_df.select(num_input).toPandas()
numeric_data1.plot(kind='box', subplots=True, layout=(2,4), sharex=False, sharey=False, figsize=(15,7))
display(plt1.show())
```


```python
#Vizualize histogram for numerical variables

import pylab as plt
numeric_data1.hist(bins=30, figsize=(14,8))
pl.suptitle("Histogram for each numeric input variable")
#plt.savefig('fruits_hist')
display(plt.show())
```


```python
#Visualize categorical variables

#Relationship between marital by deposit (Y/N) 
#notice the difference in marital between the 2 classes (graphs displayed in the following 2 blocks)

from pyspark.sql import functions as F

# show marital by deposit=yes table
marital_data_yes = spark_df.where("deposit = 'yes'").groupBy('marital').agg(F.count('marital').alias('count_yes'))
total = marital_data_yes.select("count_yes").agg(F.sum('count_yes').alias('total')).collect().pop()['total']
marital_data_yes = marital_data_yes.withColumn('ratio', (marital_data_yes['count_yes']/total)*100)
marital_data_yes.show()

# show marital by deposit=no table
marital_data_no = spark_df.where("deposit = 'no'").groupBy('marital').agg(F.count('marital').alias('count_no'))
total = marital_data_no.select("count_no").agg(F.sum('count_no').alias('total')).collect().pop()['total']
marital_data_no = marital_data_no.withColumn('ratio', (marital_data_no['count_no']/total)*100)
marital_data_no.show()

```


```python
# plot marital by deposit=yes

import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()
plt.figure(figsize=(4,3))
pandas_df = marital_data_yes.toPandas()
pandas_df.sort_values('ratio', axis = 0, ascending=False, inplace=True)
marital = pandas_df['marital']
ratio = pandas_df['ratio']
x = np.arange(len(marital))
plt.bar(x, ratio)
plt.xticks(x, marital, rotation='horizontal')
plt.title('Deposit = Yes')
display(plt.show())

```


```python
# plot marital by deposit=no

import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()
plt.figure(figsize=(4,3))
pandas_df = marital_data_no.toPandas()
pandas_df.sort_values('ratio', axis = 0, ascending=False, inplace=True)
marital = pandas_df['marital']
ratio = pandas_df['ratio']
x = np.arange(len(marital))
plt.bar(x, ratio)
plt.xticks(x, marital, rotation='horizontal')
plt.title('Deposit = No')
display(plt.show())
```


```python
# select the columns

df = spark_df.select('age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'deposit')
cols = df.columns
df.printSchema()
```


```python
#Using transformers StringIndexer, encoder, assembler

from pyspark.ml.feature import OneHotEncoderEstimator, StringIndexer, VectorAssembler
categoricalColumns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'poutcome']
stages = []
for categoricalCol in categoricalColumns:

    stringIndexer = StringIndexer(inputCol = categoricalCol, outputCol = categoricalCol + 'Index')
    encoder = OneHotEncoderEstimator(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "classVec"])
    stages += [stringIndexer, encoder]

label_stringIdx = StringIndexer(inputCol = 'deposit', outputCol = 'label')
stages += [label_stringIdx]

numericCols = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']
assemblerInputs = [c + "classVec" for c in categoricalColumns] + numericCols
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
stages += [assembler]
```


```python
# fit the pipeline 

from pyspark.ml import Pipeline
pipeline = Pipeline(stages = stages)
pipelineModel = pipeline.fit(df)
df = pipelineModel.transform(df)
selectedCols = ['label', 'features'] + cols
df = df.select(selectedCols)

```


```python
pd.DataFrame(df.take(5), columns=df.columns).transpose()
```


```python
# split the data into trainimg and testing (70/30)

train, test = df.randomSplit([0.7, 0.3], seed = 2018)
print("Training Dataset Count: " + str(train.count()))
print("Test Dataset Count: " + str(test.count()))
```


```python
# Run logistic regression on training data

from pyspark.ml.classification import LogisticRegression
lr = LogisticRegression(featuresCol = 'features', labelCol = 'label', maxIter=10)
lrModel = lr.fit(train)
```


```python
#Make predictions on the test data

predictions = lrModel.transform(test)
predictions.select('age', 'job', 'label', 'rawPrediction', 'prediction', 'probability').show(10)
```


```python
from pyspark.ml.evaluation import BinaryClassificationEvaluator
evaluator = BinaryClassificationEvaluator()
print('Test Area Under ROC', evaluator.evaluate(predictions))
```


```python
#model 1 - DecisionTree classifier

from pyspark.ml.classification import DecisionTreeClassifier
dt = DecisionTreeClassifier(featuresCol = 'features', labelCol = 'label', maxDepth = 3)
dtModel = dt.fit(train)
predictions = dtModel.transform(test)
predictions.select('age', 'job', 'label', 'rawPrediction', 'prediction', 'probability').show(10)

#evaluate performance
evaluator = BinaryClassificationEvaluator()
print("Test Area Under ROC: " + str(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})))
```


```python
#model 2 - RandomForest classifier

from pyspark.ml.classification import RandomForestClassifier
rf = RandomForestClassifier(featuresCol = 'features', labelCol = 'label')
rfModel = rf.fit(train)
predictions = rfModel.transform(test)
predictions.select('age', 'job', 'label', 'rawPrediction', 'prediction', 'probability').show(10)

#evaluate performance
evaluator = BinaryClassificationEvaluator()
print("Test Area Under ROC: " + str(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})))
```


```python
#model 3 - Gradient Boosted Tree
from pyspark.ml.classification import GBTClassifier
gbt = GBTClassifier(maxIter=10)
gbtModel = gbt.fit(train)
predictions = gbtModel.transform(test)
predictions.select('age', 'job', 'label', 'rawPrediction', 'prediction', 'probability').show(10)

#evaluate performance
evaluator = BinaryClassificationEvaluator()
print("Test Area Under ROC: " + str(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})))
```


```python
#confusion matrix 

import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix,roc_curve, auc
from sklearn import svm, datasets
import matplotlib.pyplot as plt

y = np.array(predictions.toPandas()['label'])
scores = np.array(predictions.toPandas()['prediction'])
confusion_matrix(y,scores)

```


```python
#  ROC curve plot 

import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_curve, auc

# save the predicted and probabilities in y and scores variables
y = predictions.select("label").rdd.flatMap(lambda x: x).collect()
scores = predictions.select("probability").rdd.map(lambda x: x[0][1]).collect()

fpr, tpr, thresholds = roc_curve(y, scores)
roc_auc = auc(fpr, tpr)
print(fpr)
print(tpr)
print(thresholds)
print(roc_auc)

```


```python
#ROC plot Curve vizualization

import matplotlib.pyplot as plt

plt.gcf().clear()
plt.plot(fpr, tpr, color = 'Green', label='ROC curve (area = %0.3f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
display(plt.show())
```


```python
#Predicted Deciles

import pandas as pd
import numpy as np

deciles = pd.qcut(scores, 10, labels=False)  # decile column

pred_deciles = pd.DataFrame(
    {'deposit': y,
     'Probabilities': scores,
     'Decile Number': deciles
    })

pred_deciles

```


```python
#Decile table

grouper = pred_deciles[['Decile Number', 'deposit']].groupby('Decile Number')
decile_table = grouper.count()
decile_table['Target'] = grouper.sum()['deposit']
decile_table['Nontarget'] = decile_table['deposit'] - decile_table['Target']

decile_table = decile_table.rename(index=str, columns={"deposit": "Decile Size"})
decile_table
```


```python
#cross-validation

from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
paramGrid = (ParamGridBuilder()
             .addGrid(gbt.maxDepth, [2, 4, 6])
             .addGrid(gbt.maxBins, [20, 60])
             .addGrid(gbt.maxIter, [10, 20])
             .build())
cv = CrossValidator(estimator=gbt, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=5)
# Run cross validations.This can take about 16 minutes!
cvModel = cv.fit(train)
predictions = cvModel.transform(test)
evaluator.evaluate(predictions)
```
