 #---------------(input the data)-------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sea
NK_AML=pd.DataFrame(pd.read_csv(r"your paths"))
NK_AML_sample=pd.DataFrame(pd.read_csv(r"your paths"))

#---------------(data cleaning)-----------------------------------------------------------------------
#(1)dimension
print(NK_AML.shape)
print(NK_AML)
print(NK_AML_sample)

#(2)search for all column names and recode site
NK_AML_02=pd.DataFrame(NK_AML.columns.values.tolist())
NK_AML_sample_02=pd.DataFrame(NK_AML_sample.columns.values.tolist())

#(3)rename the columns
NK_AML_03=NK_AML.rename(columns={"SAMPLE_ID": "Sample_ID"}, inplace=True)
                               
#(4)delete the the specific columns or rows from original data
#multiple columns (-1 for last , 1 is just for one position)
NK_AML_04_col=NK_AML.drop(columns=NK_AML.columns[1])

#multiple rows
NK_AML_04_row=NK_AML.drop(index=NK_AML.index[1])

#delete the missing values from the specified columns
NK_AML_00=NK_AML.dropna(subset=['AGE_01'])

#(5) select the more/less than cutoff in the specified column
NK_AML_05=NK_AML.loc[(NK_AML['AGE_02'] ==1) & (NK_AML['SEX_01'] == 1)]

#(6) append the column/rows to the original data by
#based on "specific columns" to combine them
ap_col=pd.merge(NK_AML, NK_AML_sample, on='SAMPLE_ID') #80 31

#rows (if the columns are same between the original data and the new data)
ap_row=NK_AML.append(NK_AML_sample,ignore_index=True)

#-----------(output)------------------------------------------------------------------------------
NK_AML_sample.to_csv(r"D:/Python/(2)python_coding skills/new.csv", index=True, header=True)







