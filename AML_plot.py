#------------------(data visualization)------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#input the data
NK_AML=pd.DataFrame(pd.read_csv(r"your paths"))#80
NK_AML_sample=pd.DataFrame(pd.read_csv(r"your paths"))

#Pie chart
month=np.array(NK_AML['DFS_STATUS_01'])
plt.pie(month, radius=1)
plt.show()

#Histogram plot
month=np.array(NK_AML['OS_MONTHS_01'])
plt.hist(month)
plt.title("The distribution of overall survival times in NK-AML")
plt.xlabel("Month")
plt.ylabel("Times")
plt.show()

#Scatter plot
score1=np.array(NK_AML['Stromal_score'])
score2=np.array(NK_AML['Immune_score'])
plt.scatter(score1,score2, color='black')
plt.title("The comparison between the Stromal scores and Estimate scores")
plt.xlabel("Stromal scores")
plt.ylabel("Estimate scores")
plt.show()

#Box plot 
fig, ax = plt.subplots()
age=np.array(NK_AML['AGE_02'])
score=np.array(NK_AML['Immune_score'])
plot=[age,score]
ax.boxplot(plot)
plt.title("The comparison between the age subgroup and Estimate scores")
plt.xlabel("Age subgroups")
plt.ylabel("Estimate scores")
plt.show()













