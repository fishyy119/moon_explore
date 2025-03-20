import numpy as np
from skimage.measure import block_reduce


image = np.arange(3 * 3).reshape(3, 3)  # 012 345 678
down = block_reduce(image, block_size=2, func=np.max)  # 45 78
