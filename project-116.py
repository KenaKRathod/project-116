from random import random
import pandas as pd
import plotly.express as px

df = pd.read_csv("Admission_Predict.csv")

gre_score = df["GRE Score"].tolist()
toefl_score = df["TOEFL Score"].tolist()

fig = px.scatter(x=gre_score, y=toefl_score)
fig.show()

import plotly.graph_objects as go

gre_score = df["GRE Score"].tolist()
toefl_score = df["TOEFL Score"].tolist()

admit = df["Chance of admit"].tolist()
colors=[]

for data in admit:
  if data == 1:
    colors.append("green")
  else:
    colors.append("red")



fig = go.Figure(data=go.Scatter(
    x=toefl_score,
    y=gre_score,
    mode='markers',
    marker=dict(color=colors)
))
fig.show()

score = df[["toefl_score","gre_score"]]
admit = df["Chance of admit"]

from sklearn.model_selection import train_test_split 

score_train, score_test, admit_train, admit_test = train_test_split(score, admit, test_size = 0.25, random_state = 0)
print(score_train)

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state = 0)
classifier.fit(score_train,admit_train)

[ ]


from sklearn.linear_model import LogisticRegression 

classifier = LogisticRegression(random_state = 0) 
classifier.fit(score_train, admit_train)

LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=0, solver='lbfgs', tol=0.0001, verbose=0,
                   warm_start=False)


admit_pred = classifier.predict(score_test)

from sklearn.metrics import accuracy_score 
print ("Accuracy : ", accuracy_score(admit_test, admit_pred))
