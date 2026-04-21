# Create confusion-like matrix and plot heatmap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches  # <-- NEW

models = ["ResNet", "ViT", "PyNAS", "Gaussian_NAS", "Motion_NAS", "Brightness_NAS"]
cols = ["Clean", "Gaussian Noise", "Gaussian Blur", "Brightness", "Haze"]

data = np.array([
    [0.86, 0.84, 0.86, 0.22, 0.28],
    [0.74, 0.73, 0.74, 0.05, 0.44],
    [0.86, 0.86, 0.86, 0.23, 0.26],
    [0.80, 0.82, 0.81, 0.23, 0.28],
    [0.84, 0.72, 0.84, 0.23, 0.23],
    [0.75, 0.24, 0.64, 0.72, 0.21],
])

df = pd.DataFrame(data, index=models, columns=cols)

fig, ax = plt.subplots()
cax = ax.imshow(df.values, cmap='RdYlBu')  # blue=high, red=low

# ticks
ax.set_xticks(np.arange(len(cols)))
ax.set_yticks(np.arange(len(models)))
ax.set_xticklabels(cols)
ax.set_yticklabels(models)

# move x-axis labels to the top
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')

# rotate x labels
plt.setp(ax.get_xticklabels(), rotation=45, ha="left")

# annotate
for i in range(len(models)):
    for j in range(len(cols)):
        ax.text(j, i, f"{df.values[i, j]:.2f}", ha="center", va="center")

# --------- HIGHLIGHT CELLS ---------
# Given positions (1-based indexing):
highlight_positions = [(3,1), (4,2), (5,3), (6,4)]

for (r, c) in highlight_positions:
    # convert to 0-based indexing
    i = r - 1
    j = c - 1

    rect = patches.Rectangle(
        (j - 0.5, i - 0.5),  # bottom-left corner
        1, 1,                # width, height
        linewidth=2,
        edgecolor='red',
        facecolor='none'
    )
    ax.add_patch(rect)
# -----------------------------------

# move title to bottom
ax.set_title("mIoU", y=-0.15)

plt.tight_layout()
plt.savefig('cm.png')
plt.show()