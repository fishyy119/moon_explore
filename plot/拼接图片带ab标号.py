from plot_utils import *
from PIL import Image
import matplotlib.gridspec as gridspec
import io


# ---- Step 1: 输入图片路径 ----
n = int(input("图片数量："))
img_paths: List[Path] = [Path(input(f"第{i+1}张图片路径：")) for i in range(n)]

# ---- Step 2: 获取图片宽高比 ----
aspect_ratios = []
images = []

for path in img_paths:
    if "svg" in path.suffix:
        import cairosvg

        # * need https://github.com/tschoonj/GTK-for-Windows-Runtime-Environment-Installer
        png_bytes = cairosvg.svg2png(url=str(path), dpi=3000)
        img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")  # type: ignore
    else:
        img = Image.open(path)
    images.append(np.array(img))  # 保存为 numpy 数组用于 imshow
    width, height = img.size
    aspect_ratios.append(width / height)

fig = plt.figure()
# ==================================================================
# 两种方案，等宽 / 等高
# gs = gridspec.GridSpec(1, n, width_ratios=aspect_ratios)
gs = gridspec.GridSpec(1, n, width_ratios=[1 for _ in range(n)])
# ==================================================================
axes = [plt.subplot(gs[i]) for i in range(n)]

for ax, img in zip(axes, images):
    ax.imshow(img)
    ax.axis("off")
    ax.set_anchor("S")

# ---- Step 4: 添加 ABC 编号（如需要） ----
axes_add_abc(axes)
plt_tight_show()

# 如需保存：
fig.savefig("aligned_output.png", dpi=1000)
