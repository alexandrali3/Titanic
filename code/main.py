
# coding: utf-8

# In[346]:


get_ipython().magic(u'matplotlib inline')


# In[347]:


import sys
import numpy as np
import pandas as pd
import re as re
import matplotlib.pyplot as plt


# In[348]:


train = pd.read_csv('../data/train.csv', header = 0, dtype={'Age': np.float64})
test = pd.read_csv('../data/test.csv', header = 0, dtype={'Age': np.float64})
full_data = [train, test]


# In[349]:


print (train.info())


# In[350]:


train.describe()


# In[351]:


train.Survived.value_counts().plot(kind='bar')
plt.title(u"survived (1:survived)") # 标题
plt.ylabel(u"people")


# In[352]:


train.Pclass.value_counts(sort = False).plot(kind='bar')
plt.title(u"people")
plt.ylabel(u"cabin_level")


# In[353]:


plt.scatter(train.Survived, train.Age)
plt.ylabel(u"age")
plt.grid(b = True, which = 'major', axis = 'y')
plt.title(u"age-survived (1:survived)")


# In[354]:


train.Age[train.Pclass == 1].plot(kind='kde') # Kernel Density Estimation
train.Age[train.Pclass == 2].plot(kind='kde')
train.Age[train.Pclass == 3].plot(kind='kde')
plt.xlabel(u"age")
plt.ylabel(u"p")
plt.title(u"cabin_level-age")
plt.legend((u'level_0', u'level_1', u'level_2'), loc = 'best')


# In[355]:


train.Embarked.value_counts().plot(kind='bar')
plt.title(u"embarked")
plt.ylabel(u"people")
plt.show()


# In[356]:


Survived_0 = train.Pclass[train.Survived == 0].value_counts()
Survived_1 = train.Pclass[train.Survived == 1].value_counts()
df = pd.DataFrame({u'survived':Survived_1, u'not survived':Survived_0})
df.plot(kind='bar', stacked=True)
plt.title(u"cabin_level-survived")
plt.xlabel(u"level")
plt.ylabel(u"people")
plt.show()


# In[357]:


Survived_m = train.Survived[train.Sex == 'male'].value_counts()
Survived_f = train.Survived[train.Sex == 'female'].value_counts()
df = pd.DataFrame({u'male':Survived_m, u'female':Survived_f})
df.plot(kind='bar', stacked=True)
plt.title(u"sex-survived")
plt.xlabel(u"sex")
plt.ylabel(u"people")
plt.show()


# In[358]:


fig = plt.figure()

ax1 = fig.add_subplot(141)
train.Survived[train.Sex == 'female'][train.Pclass != 3].value_counts().plot(kind='bar', label="female, high class")
ax1.set_xticklabels([u"not", u"survived"], rotation = 0)
ax1.legend([u"female-high"], loc='best')

ax2 = fig.add_subplot(142, sharey=ax1)
train.Survived[train.Sex == 'female'][train.Pclass == 3].value_counts().plot(kind='bar', label="female, low class")
ax2.set_xticklabels([u"not", u"survived"], rotation = 0)
ax2.legend([u"female-low"], loc='best')

ax3 = fig.add_subplot(143, sharey=ax1)
train.Survived[train.Sex == 'male'][train.Pclass != 3].value_counts().plot(kind='bar', label="male, high class")
ax3.set_xticklabels([u"not", u"survived"], rotation = 0)
ax3.legend([u"male-high"], loc='best')

ax4 = fig.add_subplot(144, sharey=ax1)
train.Survived[train.Sex == 'male'][train.Pclass == 3].value_counts().plot(kind='bar', label="male, low class")
ax4.set_xticklabels([u"not", u"survived"], rotation = 0)
ax4.legend([u"male-low"], loc='best')

plt.show()


# # SibSp and Parch

# In[359]:


for dataset in full_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
print (train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index = False).mean())


# In[360]:


for dataset in full_data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
print (train[['IsAlone', 'Age', 'Survived']].groupby(['IsAlone'], as_index = False).mean())


# # Embarked

# In[361]:


for dataset in full_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
print (train[['Embarked', 'Survived']].groupby(['Embarked'], as_index = False).mean())


# # Fare

# In[362]:


for dataset in full_data:
    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())
train['CategoricalFare'] = pd.qcut(train['Fare'], 4)
print (train[['CategoricalFare', 'Survived']].groupby(['CategoricalFare'], as_index = False).mean())


# # Age

# In[363]:


for dataset in full_data:
    age_avg = dataset['Age'].mean()
    age_std = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()
    
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)
    
train['CategoricalAge'] = pd.cut(train['Age'], 5)

print (train[['CategoricalAge', 'Survived']].groupby(['CategoricalAge'], as_index = False).mean())


# # Name

# In[364]:


def get_title(name):
    title_search = re.search('([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""

for dataset in full_data:
    dataset['Title'] = dataset['Name'].apply(get_title)
    
print (pd.crosstab(train['Title'], train['Sex']))


# In[365]:


for dataset in full_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt', 'Col',                                                'Don', 'Dr', 'Major', 'Rev', 'Sir',                                                'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Miss')

print (train[['Title', 'Survived']].groupby(['Title'], as_index = False).mean())


# In[366]:


for dataset in full_data:
    # Mopping Sex
    dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
    
    # Mopping titles
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)
    
    # Mopping Embarked
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
    
    # Mopping Fare
    dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
    dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)
    
    # Mopping Age
    dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[dataset['Age'] > 64, 'Age'] = 4

# Feature Selection
drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp']
train = train.drop(drop_elements, axis = 1)
train = train.drop(['CategoricalAge', 'CategoricalFare'], axis = 1)

id_series = pd.Series(test["PassengerId"])
test = test.drop(drop_elements, axis = 1)

print (train.head(10))

train_pd, test_pd = train, test

train = train.drop(['Parch', 'FamilySize'], axis = 1)
test = test.drop(['Parch', 'FamilySize'], axis = 1)

train = train.values
test = test.values


# In[367]:


print(train)


# # Classifier Comparison

# In[368]:


import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression


# In[369]:


classifiers = [
    KNeighborsClassifier(3),
    SVC(probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),
    LogisticRegression()
]

log_cols = ["Classifier", "Accuracy"]
log = pd.DataFrame(columns=log_cols)

sss = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=0)

X = train[0::, 1::]
y = train[0::, 0]

acc_dict = {}


# In[370]:


for train_index, test_index in sss.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # print (y_train.shape, y_test.shape)
    
    for clf in classifiers:
        name = clf.__class__.__name__
        clf.fit(X_train, y_train)
        train_predictions = clf.predict(X_test)
        acc = accuracy_score(y_test, train_predictions)
        if name in acc_dict:
            acc_dict[name] += acc
        else:
            acc_dict[name] = acc
        
for clf in acc_dict:
    acc_dict[clf] = acc_dict[clf] / 10.0
    log_entry = pd.DataFrame([[clf, acc_dict[clf]]], columns=log_cols)
    # print (log_entry)
    log = log.append(log_entry)

plt.xlabel('Accuracy')
plt.title('Classifier Accuracy')

sns.set_color_codes("muted")
sns.barplot(x='Accuracy', y='Classifier', data=log, color="b")


# In[371]:


candidate_classifier = SVC()
candidate_classifier.fit(train[0::, 1::], train[0::, 0])
result = candidate_classifier.predict(test)


# In[372]:


print(result)


# In[373]:


result_df = pd.DataFrame({"PassengerId": id_series, "Survived": result})
result_df.to_csv('../data/result.csv', index=False)


# In[374]:


colormap = plt.cm.viridis
plt.figure(figsize=(12, 12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(train_pd.astype(float).corr(), linewidths=0.1, vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
plt.show()


# In[375]:


g = sns.pairplot(train_pd[[u'Survived', u'Pclass', u'Sex', u'Age', u'Parch', u'Fare', u'Embarked',                          u'FamilySize', u'Title']], hue='Survived', palette = 'seismic', size=1.2, diag_kind =                 'kde', diag_kws=dict(shade=True), plot_kws=dict(s=10) )
g.set(xticklabels=[])


# In[377]:


train_pd.shape[0]

