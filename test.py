import pandas as pd
import matplotlib.pyplot as plt
file_name = 'fashionmnist_result_my.csv'
df = pd.read_csv(file_name, index_col=0)
# df = df.iloc[:, 0:5]
n = df.shape[0]

title = 'FixMatch-FashionMNIST-Noise'
plt.title(title)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
x_axis = list(range(n))
for col in df:
    plt.plot(x_axis, df[col], label=col)

plt.legend()
plt.savefig(title+'.png')
