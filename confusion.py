import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
# Adjusting the confusion matrix and labels as per the new requirements

matplotlib.use('Agg')


cm_custom = np.array([[555, 45], [8, 592]])
labels = ["Non-Hotspot", "Hotspot"]

# Plot customized confusion matrix
#plt.rcParams.update({'font.family':'Helvetica'})
#plt.rcParams.update({'font.family':'sans-serif'})

# Re-plotting the confusion matrix with the specified font
plt.figure(figsize=(12,9))
plt.tight_layout()
#plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)
ax = sns.heatmap(cm_custom, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels,
                 annot_kws={"size": 20}
                 )
# 调整色标的标签字体大小
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=20)
plt.xticks(fontsize=20, rotation=0)
plt.yticks(fontsize=20, rotation=90)
plt.ylabel('True label', fontsize=20)
plt.xlabel('Predicted label', fontsize=20)
plt.savefig('./model/混淆矩阵.png')



