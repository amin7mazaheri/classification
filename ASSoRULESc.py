#Packages that you need: pandas, numpy, mlxtend
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori,association_rules
df = pd.read_csv('D:/Laptop backup 2/course work2/course works2/IUST/stat course/lectures/files/service5.csv')
df.describe(include="all").iloc[:,:3]
np.sum(df.isna())
D=df.dropna()
A=pd.get_dummies(D)
frequent_itemsets = apriori(A, min_support=0.05, use_colnames=True,max_len=3)
rules = association_rules(frequent_itemsets, metric="confidence",min_threshold=0.2)
#'''
#Get some info about time-related rules
Month=A.columns[0:6]
Monthconseq=[[x,rules[rules['antecedents']=={x}].loc[:,['consequents','support','confidence','lift','leverage','conviction']]] for x in A.columns[0:6].tolist() if np.sum(rules['antecedents']=={x})>0]
Monthconseq[0][1].iloc[:,:2]
Monthconseq[0][1].iloc[:,2:4]

Monthconseq[1][0]
Monthconseq[1][1].iloc[:,:2]
Monthconseq[1][1].iloc[:,2:4]

#Get some info about agency-related rules
Agency=A.columns[6:23]
Agencyconseq=[[x,rules[rules['antecedents']=={x}].loc[:,['consequents','support','confidence','lift','leverage','conviction']]] for x in A.columns[6:23].tolist() if np.sum(rules['antecedents']=={x})>0]
Agencyconseq[0][0]
Agencyconseq[0][1].iloc[:,0:5]
Agencyconseq[0][1].iloc[:,5]
Agencyconseq[1][0]
Agencyconseq[1][1].iloc[:,:5]
Agencyconseq[1][1].iloc[:,5]

#Get some info about borough-related rules
Borough=A.columns[1091:]
Boroughconseq=[[x,rules[rules['antecedents']=={x}].loc[:,['consequents','support','confidence','lift','leverage','conviction']]] for x in A.columns[1091:].tolist() if np.sum(rules['antecedents']=={x})>0]

#Get some info about complaint-type-related rules
Complaint=A.columns[23:162]
Complaintconseq=[[x,rules[rules['antecedents']=={x}].loc[:,['consequents','support','confidence','lift','leverage','conviction']]] for x in A.columns[23:162].tolist() if np.sum(rules['antecedents']=={x})>0]

Complaintconseq[2][1].iloc[:,:2]
Complaintconseq[2][1].iloc[:,2:6]

#Hypothesis Test

phatsd=np.sqrt(np.mean(rules['support'])*(1-np.mean(rules['support']))/len(rules['support']))
phatbar=np.mean(rules['support'])
z=0.045/phatsd
from scipy.stats import norm
norm.isf(1-0.975)
#'''
#Removing NYPD and HPD
D2=D[D['Agency']!='NYPD']
D2=D2[D2['Agency']!='HPD']
A2=pd.get_dummies(D2)
frequent_itemsets2 = apriori(A2, min_support=0.05, use_colnames=True,max_len=2)
rules2 = association_rules(frequent_itemsets2, metric="confidence",min_threshold=0.2)

phatsd2=np.sqrt(np.mean(rules2['support'])*(1-np.mean(rules2['support']))/len(rules2['support']))
phatbar2=np.mean(rules2['support'])
z=0.065/phatsd2

#Get some info about time-related rules
Month2=A2.columns[0:6]
Monthconseq2=[[x,rules2[rules2['antecedents']=={x}].loc[:,['consequents','support','confidence','lift','leverage','conviction']]] for x in A2.columns[:6].tolist() if np.sum(rules2['antecedents']=={x})>0]
Monthconseq2[3][1].iloc[:,0]
Monthconseq2[3][1].iloc[:,1:]

Monthconseq2[5][1].iloc[:,0]
Monthconseq2[5][1].iloc[:,1:]
#Get some info about agency-related rules
Agency2=A2.columns[6:21]
Agencyconseq2=[[x,rules2[rules2['antecedents']=={x}].loc[:,['consequents','support','confidence','lift','leverage','conviction']]] for x in A2.columns[6:21].tolist() if np.sum(rules2['antecedents']=={x})>0]
Agencyconseq2[0][1].iloc[:,1:]

#Get some info about borough-related rules
Borough2=A2.columns[910:]
Boroughconseq2=[[x,rules2[rules2['antecedents']=={x}].loc[:,['consequents','support','confidence','lift','leverage','conviction']]] for x in A2.columns[910:].tolist() if np.sum(rules2['antecedents']=={x})>0]
Boroughconseq2[2][1].iloc[:,0]
Boroughconseq2[2][1].iloc[:,1:]

#Get some info about complaint-type-related rules
Complaint2=A2.columns[21:125]
Complaintconseq2=[[x,rules2[rules2['antecedents']=={x}].loc[:,['consequents','support','confidence','lift','leverage','conviction']]] for x in A2.columns[21:125].tolist() if np.sum(rules2['antecedents']=={x})>0]
Complaintconseq2[0][1].iloc[:,0]
Complaintconseq2[0][1].iloc[:,1:]
#'''
