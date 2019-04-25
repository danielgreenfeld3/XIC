import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#load the losses
mse_exp, hsic_exp, mae_exp = [],[],[]
for i in [1,2,3]:
    mse_exp.append(np.load('exp/non_linear_32mse'+str(i)+'.npy'))
    hsic_exp.append(np.load('exp/non_linear_32hsic' + str(i) + '.npy'))
    mae_exp.append(np.load('exp/non_linear_32mae' + str(i) + '.npy'))

#arrange the losses in a dataframe object which contains information on the training loss and the test city
dfs_mse = pd.concat([pd.DataFrame({'RMSE':mse_exp[i],
                                   'Training loss':['mse' for _ in range(len(mse_exp[i]))],
                                   'test city':[i+1 for _ in range(len(mse_exp[i]))]}) for i in range(3)])
dfs_hsic = pd.concat([pd.DataFrame({'RMSE':hsic_exp[i],
                                    'Training loss':['hsic' for _ in range(len(hsic_exp[i]))],
                                    'test city':[i+1 for _ in range(len(hsic_exp[i]))]}) for i in range(3)])
dfs_mae = pd.concat([pd.DataFrame({'RMSE':mae_exp[i],
                                   'Training loss':['mae' for _ in range(len(mae_exp[i]))],
                                   'test city':[i+1 for _ in range(len(mae_exp[i]))]}) for i in range(3)])
df = pd.concat([dfs_mse, dfs_hsic,dfs_mae])

#create a box plot
g=sns.boxplot(x='test city', y='RMSE',data=df,hue='Training loss')
sns.set(rc={'figure.figsize':(13,10)})
plt.show()