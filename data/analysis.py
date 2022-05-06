import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import pandas as pd
from scipy.stats import norm
from sklearn.model_selection import train_test_split
import seaborn as sns

dataset = pd.read_csv('./ingotRate.csv')
# data = dataset.iloc[:, :]

# all parameters distribution in ingot rate
fig, ax = plt.subplots(nrows=3, ncols=int(dataset.shape[1]/3), figsize=(20, 12))
for i in range(3):
    for j in range(int(ax.size/3)):
        curLine=dataset.iloc[:, i*int(dataset.shape[1]/3)+j]
        ax[i][j].hist(curLine, 50, density=1,alpha=0.75, label=curLine.name)
        lines, labels = ax[i][j].get_legend_handles_labels()
        ax[i][j].legend(lines, labels, loc='best')
fig.suptitle('Ingot - Parameters distribution', y=0.92, fontsize=20)
plt.show()
plt.savefig("ingotRateParametersDistribution.png",bbox_inches='tight')

# label distribution in ingot rate
# X = dataset.iloc[:, :-1]
# y = dataset.iloc[:, dataset.shape[1] - 1]
# X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2, random_state=3)
# fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
# ax[0].hist(y_train, 50, density=1,alpha=0.75, label='y_train')
# lines, labels = ax[0].get_legend_handles_labels()
# ax[0].legend(lines, labels, loc='best')
# ax[1].hist(y_test, 50, density=1,alpha=0.75, label='y_test')
# lines, labels = ax[1].get_legend_handles_labels()
# ax[1].legend(lines, labels, loc='best')
# fig.suptitle('Ingot - Labels distribution', y=0.95, fontsize=12)
# # plt.show()
# plt.savefig("ingotRateLabelsDistribution.png",bbox_inches='tight')




y = dataset.iloc[:, dataset.shape[1] - 1]
fig, ax = plt.subplots()
ax.set(xlabel='Ingot Rate',
       ylabel='Distribution',
       title='Ingot Rate - Labels Distribution')
ax.hist(y, 50, density=1, alpha=0.75, label='y_train', color='#9EABD9')
sns.kdeplot(y, color='#223B8F', shade=True, linestyle="--", linewidth=3)
lines, labels = ax.get_legend_handles_labels()
ax.legend(lines, labels, loc='best')
# plt.show()
plt.savefig("ingotRateLabelsDistribution-all.png",bbox_inches='tight')










dataset = pd.read_csv('./yieldRate.csv')

# all parameters distribution in yield rate
# data = dataset.iloc[:, :]
#
# fig, ax = plt.subplots(nrows=2, ncols=int(dataset.shape[1]/2), figsize=(12, 8))
# for i in range(2):
#     for j in range(int(ax.size/2)):
#         curLine=dataset.iloc[:, i*int(dataset.shape[1]/2)+j]
#         ax[i][j].hist(curLine, 50, density=1,alpha=0.75, label=curLine.name)
#         lines, labels = ax[i][j].get_legend_handles_labels()
#         ax[i][j].legend(lines, labels, loc='best')
# fig.suptitle('Yield - Parameters distribution', y=0.93, fontsize=20)
# plt.show()
# plt.savefig("yieldRateParametersDistribution.png",bbox_inches='tight')


# label distribution in yield rate
# plt.style.use('bmh')
# X = dataset.iloc[:, :-1]
# y = dataset.iloc[:, dataset.shape[1] - 1]
# X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2, random_state=0)
# fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
# axResult=ax[0].hist(y_train, 30, density=1,alpha=0.75, label='y_train')
# lines, labels = ax[0].get_legend_handles_labels()
# ax[0].legend(lines, labels, loc='best')
# ax[1].hist(y_test, 50, density=1,alpha=0.75, label='y_test')
# sns.kdeplot(y_test ,color='#0AA7B0',shade=True,linestyle="--",linewidth=2)
# lines, labels = ax[1].get_legend_handles_labels()
# ax[1].legend(lines, labels, loc='best')
# fig.suptitle('Yield - Labels distribution', y=0.95, fontsize=12)
# plt.show()
# plt.savefig("yieldRateLabelsDistribution.png",bbox_inches='tight')


# y = dataset.iloc[:, dataset.shape[1] - 1]
# fig, ax = plt.subplots()
# ax.set(xlabel='Yield Rate',
#        ylabel='Distribution',
#        title='Yield Rate - Labels Distribution')
# ax.hist(y, 50, density=1, alpha=0.75, label='y_train', color='#DBA89B')
# sns.kdeplot(y, color='#8F361F', shade=True, linestyle="--", linewidth=3)
# lines, labels = ax.get_legend_handles_labels()
# ax.legend(lines, labels, loc='best')
# # plt.show()
# plt.savefig("yieldRateLabelsDistribution-all.png",bbox_inches='tight')