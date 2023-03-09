#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

import seaborn as sns


# In[3]:


data1 = pd.read_excel("~/Desktop/data/wtc data/merged files/Data Request 07-06-2018_SBU_MSSM.xlsx")


# In[4]:


data1 = data1.rename(columns={"StudyIDnew":"Joint_ID"})


# In[5]:


data1


# In[6]:


men = data1[data1['Gender']==0].copy()
women = data1[data1['Gender']==1].copy()


# In[7]:


men


# In[8]:


women


# In[9]:


BMI_w = (women['weight_lbs']/2.205)/(women['hight_in']/39.37)
Spline1_w = (-1*((women['Age'] - 8)**3)*(74 - 13))/((74-8) +(women['Age']-13)**3)
Spline2_w = (-1*((women['Age'] - 8)**3)*(74 - 19))/((74-8) +(women['Age']-19)**3)


# In[10]:


Spline1_w


# In[11]:


women['predicted_FEV1'] = 2.714397 + 0.0852553*BMI_w + (-0.0015578)*BMI_w**2 + (-0.0710514)*(women['hight_in']/39.37)+0.0003158*(women['hight_in']/39.37)**2 + 0.2189869*women['Age'] + 0.0025242*Spline1_w + (-0.0011336)*Spline2_w


# In[12]:


women


# In[13]:


BMI_m = (men['weight_lbs']/2.205)/(men['hight_in']/39.37)**2
Spline1_m = ((-1*(men['Age'] - 8)**3)*(74 - 21))/(74-8) +(men['Age']-21)**3
Spline2_m = ((-1*(men['Age'] - 8)**3)*(74 - 25))/(74-8) +(men['Age']-25)**3


# In[14]:


BMI_m


# In[15]:


Spline1_m


# In[16]:


men['predicted_FEV1'] = 5.43862 + 0.1918785*BMI_m + (-0.0034633)*BMI_m**2 + (-0.1241893)*(men['hight_in']*2.54)+0.0005119*(men['hight_in']*2.54)**2 + 0.1603364 *men['Age'] + 0.0011688*Spline1_m + (-0.0008873)*Spline2_m


# In[17]:


men


# In[18]:


men['FEV_1sec']


# In[19]:


test_BMI = 20.19
test_spline = ((-1*(32 - 8)**3)*(74 - 21))/(74-8) +(32-21)**3
test_spline_2 = ((-1*(32 - 8)**3)*(74 - 25))/(74-8) +(32-25)**3


# In[20]:


test_prediction = 5.43862 + 0.1918785*test_BMI + (-0.0034633)*test_BMI**2 + (-0.1241893)*(168)+0.0005119*(168)**2 + 0.1603364 *32 + 0.0011688*test_spline + (-0.0008873)*test_spline_2


# In[21]:


test_prediction


# In[22]:


BMI_w = (women['weight_lbs']/2.205)/(women['hight_in']/39.37)
Spline1_w = (-1*((women['Age'] - 8)**3)*(74 - 13))/((74-8) +(women['Age']-13)**3)
Spline2_w = (-1*((women['Age'] - 8)**3)*(74 - 19))/((74-8) +(women['Age']-19)**3)


# In[23]:


BMI_w


# In[24]:


BMI_w = (women['weight_lbs']/2.205)/(women['hight_in']/39.37)**2
Spline1_w = ((-1*(women['Age'] - 8)**3)*(74 - 13))/(74-8) +(women['Age']-13)**3
Spline2_w = ((-1*(women['Age'] - 8)**3)*(74 - 19))/(74-8) +(women['Age']-19)**3


# In[25]:


BMI_w


# In[26]:


Spline1_w


# In[27]:


Spline2_w


# In[28]:


women['predicted_FEV1'] = 2.714397 + 0.0852553*BMI_w + (-0.0015578)*BMI_w**2 + (-0.0710514)*(women['hight_in']*2.54)+0.0003158*(women['hight_in']*2.54)**2 + 0.2189869*women['Age'] + 0.0025242*Spline1_w + (-0.0011336)*Spline2_w


# In[29]:


women


# In[30]:


men['∆FEV1'] = men['predicted_FEV1'] - men['FEV_1sec']
women['∆FEV1'] = women['predicted_FEV1'] - women['FEV_1sec']


# In[31]:


men


# In[32]:


men = men.drop([54])


# In[33]:


men


# In[34]:


women


# In[35]:


instability_wtc_data = pd.read_csv("~/Desktop/data/wtc data/wtc_instability.csv")
instability_wtc_data = instability_wtc_data.rename(columns={"0":"instability"})
instability_wtc_data


# In[36]:


inst_men = pd.merge(men, instability_wtc_data, on="Joint_ID")
inst_women = pd.merge(women, instability_wtc_data, on= "Joint_ID")


# In[37]:


inst_men


# In[38]:


inst_women


# In[39]:


inst_women['∆FEV1']


# In[40]:


inst_women.plot.scatter(x='instability',y='∆FEV1')


# In[41]:


inst_men.plot.scatter(x='∆FEV1', y='instability')


# In[42]:


wtc_joined = pd.concat([inst_women, inst_men])


# In[43]:


wtc_joined


# In[44]:


wtc_joined.plot.scatter(x='instability', y='∆FEV1')


# In[45]:


sns.violinplot(data=wtc_joined, y="∆FEV1", x="instability")


# In[46]:


x= wtc_joined['instability']
y= wtc_joined['∆FEV1']

a, b = np.polyfit(x, y, 1)
plt.scatter(x, y)
plt.plot(x, a*x+b)


plt.axhline(y=0, color='r', linestyle='-')
plt.show()


# In[47]:


data2 = pd.read_csv("~/Desktop/data/wtc data/wtc_ins_age_regressed_out.csv")


# In[48]:


data2


# In[49]:


data2 = data2.rename(columns={"ID":"Joint_ID"})


# In[50]:


wtc_joined["instability"] = data2["ins_age_regressed_out"]#check that they are joined at the key
data3 = data2[["Joint_ID", "ins_age_regressed_out", "age_regression"]]


# In[51]:


data3


# In[52]:


data4 = wtc_joined.merge(data3, on="Joint_ID")


# In[53]:


wtc_joined


# In[54]:


data4


# In[55]:


wtc_joined[["Joint_ID", "instability"]]


# In[56]:


wtc_joined.plot.scatter(x='instability', y='∆FEV1')
plt.axhline(y=0, color='r', linestyle='-')
plt.show()


# In[57]:


data4.plot.scatter(x='instability', y='∆FEV1')
plt.axhline(y=0, color='r', linestyle='-')
plt.show()


# In[58]:


data2.plot.scatter(x='ins_age_regressed_out', y='age')
data2.plot.scatter(x='age_regression', y='age')


# In[59]:


x = data4["ins_age_regressed_out"]
y = data4["∆FEV1"]

a, b = np.polyfit(x, y, 1)
plt.scatter(x, y)
plt.plot(x, a*x+b)

plt.axhline(y=0, color='r', linestyle='-')
plt.show()


# In[60]:


data2["age_bucket"]


# In[61]:


x1 = data4["ins_age_regressed_out"]
y1 = data4["∆FEV1"]
x2 = wtc_joined["instability"]
y2 = wtc_joined["∆FEV1"]

a, b = np.polyfit(x1, y1, 1)
plt.scatter(x1, y1)
plt.plot(x1, a*x1+b)

plt.axhline(y=0, color='r', linestyle='-')
plt.show()


# In[62]:


data4


# In[63]:


data4["expected_instability"] = data4["age_regression"] + 0.5110


# In[64]:


data4


# In[65]:


data4[["instability", "expected_instability"]]


# In[66]:


x1 = data4["instability"]
y1 = data4["∆FEV1"]
x2 = data4["expected_instability"]

a, b = np.polyfit(x1, y1, 1)
plt.scatter(x1, y1)
plt.plot(x1, a*x1+b)

a, b = np.polyfit(x2, y1, 1)
plt.scatter(x2, y1)
plt.plot(x2, a*x2+b)

plt.axhline(y=0, color='r', linestyle='-')
plt.show()


# In[67]:


data4.plot.scatter(x='expected_instability',y='Age')


# In[68]:


data4.plot.scatter(x='expected_instability',y="∆FEV1")

a, b = np.polyfit(x, y, 1)
plt.scatter(x, y)
plt.plot(x, a*x+b)


plt.axhline(y=0, color='r', linestyle='-')
plt.show()


# In[69]:


expected_instability_given_exposure = data4[data4["Pt_Type"]==1]
expected_instability_given_exposure


# In[70]:


expected_instability_given_exposure.plot.scatter(x='ins_age_regressed_out',y="∆FEV1")

a, b = np.polyfit(x, y, 1)
plt.scatter(x, y)
plt.plot(x, a*x+b)


plt.axhline(y=0, color='r', linestyle='-')
plt.show()


# In[71]:


actual_instability = data4[data4["Pt_Type"]!=1]
actual_instability


# In[72]:


x1 = expected_instability_given_exposure["ins_age_regressed_out"]
y1 = expected_instability_given_exposure["∆FEV1"]
x2 = actual_instability["ins_age_regressed_out"]
y2 = actual_instability["∆FEV1"]

a, b = np.polyfit(x1, y1, 1)
plt.scatter(x1, y1)
plt.plot(x1, a*x1+b)

a, b = np.polyfit(x2, y2, 1)
plt.scatter(x2, y2)
plt.plot(x2, a*x2+b)

plt.axhline(y=0, color='r', linestyle='-')
plt.show()


# In[73]:


x1 = expected_instability_given_exposure["ins_age_regressed_out"]
y1 = expected_instability_given_exposure["∆FEV1"]
x2 = actual_instability["ins_age_regressed_out"]
y2 = actual_instability["∆FEV1"]
x3 = actual_instability["expected_instability"]
x4 = expected_instability_given_exposure["expected_instability"]

a, b = np.polyfit(x1, y1, 1)
plt.plot(x1, a*x1+b)

a, b = np.polyfit(x2, y2, 1)
plt.plot(x2, a*x2+b)

a, b = np.polyfit(x4, y1, 1)
plt.scatter(x4, y1)
plt.plot(x3, a*x3+b)

a, b = np.polyfit(x3, y2, 1)
plt.scatter(x3, y2)
plt.plot(x3, a*x3+b)

plt.axhline(y=0, color='r', linestyle='-')
plt.show()


# In[74]:


x1 = expected_instability_given_exposure["ins_age_regressed_out"]
y1 = expected_instability_given_exposure["∆FEV1"]
x2 = actual_instability["ins_age_regressed_out"]
y2 = actual_instability["∆FEV1"]

a, b = np.polyfit(x1, y1, 1)
plt.plot(x1, a*x1+b)

a, b = np.polyfit(x2, y2, 1)
plt.plot(x2, a*x2+b)

plt.axhline(y=0, color='r', linestyle='-')
plt.show()


# In[75]:


x1 = expected_instability_given_exposure["ins_age_regressed_out"]
y1 = expected_instability_given_exposure["∆FEV1"]

a, b = np.polyfit(x1, y1, 1)
plt.plot(x1, a*x1+b)

plt.axhline(y=0, color='r', linestyle='-')
plt.show()


# In[76]:


import seaborn as sns


# In[77]:


sns.regplot(
 data=data4, x="ins_age_regressed_out", y="∆FEV1",
)


# In[79]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm


# In[81]:


X = data4["ins_age_regressed_out"] 
y = data4["∆FEV1"]

X2 = sm.add_constant(X)
est = sm.OLS(y, X2)
est2 = est.fit()
print(est2.summary())


# In[82]:


sns.regplot(
 data=expected_instability_given_exposure, x="ins_age_regressed_out", y="∆FEV1",
)
sns.regplot(
 data=actual_instability, x="ins_age_regressed_out", y="∆FEV1",
)

plt.axhline(y=0, color='r', linestyle='-')
plt.show()


# In[83]:


X = expected_instability_given_exposure["ins_age_regressed_out"] 
y = expected_instability_given_exposure["∆FEV1"]

X2 = sm.add_constant(X)
est = sm.OLS(y, X2)
est2 = est.fit()
print(est2.summary())


# In[84]:


X = actual_instability["ins_age_regressed_out"] 
y = actual_instability["∆FEV1"]

X2 = sm.add_constant(X)
est = sm.OLS(y, X2)
est2 = est.fit()
print(est2.summary())


# In[85]:


data4


# In[88]:


control = data4[data4['Pt_Type']==1]
CI = data4[data4['Pt_Type']==2]
PTSD = data4[data4['Pt_Type']==3]
CI_PTSD = data4[data4['Pt_Type']==4]


# In[89]:


control


# In[90]:


CI


# In[92]:


PTSD


# In[93]:


CI_PTSD


# In[94]:


data4["∆FEV1"]


# In[96]:


sns.regplot(
 data=control, x="ins_age_regressed_out", y="∆FEV1",
)
sns.regplot(
 data=CI, x="ins_age_regressed_out", y="∆FEV1",
)
sns.regplot(
 data=PTSD, x="ins_age_regressed_out", y="∆FEV1",
)
sns.regplot(
 data=CI_PTSD, x="ins_age_regressed_out", y="∆FEV1",
)
plt.axhline(y=0, color='r', linestyle='-')
plt.show()


# In[97]:


sns.regplot(
 data=control, x="ins_age_regressed_out", y="∆FEV1",
)
plt.axhline(y=0, color='r', linestyle='-')
plt.show()


# In[98]:


sns.regplot(
 data=CI, x="ins_age_regressed_out", y="∆FEV1",
)


# In[99]:


sns.regplot(
 data=PTSD, x="ins_age_regressed_out", y="∆FEV1",
)


# In[100]:


sns.regplot(
 data=CI_PTSD, x="ins_age_regressed_out", y="∆FEV1",
)


# In[101]:


X = control["ins_age_regressed_out"] 
y = control["∆FEV1"]

X2 = sm.add_constant(X)
est = sm.OLS(y, X2)
est2 = est.fit()
print(est2.summary())


# In[102]:


X = CI["ins_age_regressed_out"] 
y = CI["∆FEV1"]

X2 = sm.add_constant(X)
est = sm.OLS(y, X2)
est2 = est.fit()
print(est2.summary())


# In[103]:


X = PTSD["ins_age_regressed_out"] 
y = PTSD["∆FEV1"]

X2 = sm.add_constant(X)
est = sm.OLS(y, X2)
est2 = est.fit()
print(est2.summary())


# In[105]:


X = CI_PTSD["ins_age_regressed_out"] 
y = CI_PTSD["∆FEV1"]

X2 = sm.add_constant(X)
est = sm.OLS(y, X2)
est2 = est.fit()
print(est2.summary())


# In[ ]:




