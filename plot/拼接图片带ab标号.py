from plot_utils import *

fig, axes = plt.subplots(1, 2)
for ax in axes:
    img_path = input("位图路径：")
    img = plt.imread(img_path)
    ax.imshow(img)
    ax.set_anchor("S")
    ax.axis("off")

axes_add_abc(axes)
plt_tight_show()
