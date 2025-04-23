import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Initialize a 501x501 numpy array with zeros
array = np.zeros((501, 501), dtype=int)

# Calculate the angle in radians for each point and assign values based on the angle
center_x, center_y = 250, 250
for i in range(501):
    for j in range(501):
        dx, dy = j - center_x, i - center_y
        angle = (np.degrees(np.arctan2(dy, dx)) + 360) % 360  # Angle in radians [0, 2π)
        if 0 <= angle < 135:  # 0 to 135 degrees
            array[i, j] = 0
        elif 135 <= angle < 270:  # 135 to 270 degrees
            array[i, j] = 1
        else:  # 270 to 360 degrees
            array[i, j] = 2

# Save or use the array as needed
# Display the array using imshow
plt.imshow(array, cmap="viridis", origin="lower")
plt.colorbar(label="Region Value")
plt.title("Region Map Visualization")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()


NPY_ROOT = Path(__file__).parent.parent / "resource"
np.save(NPY_ROOT / "map_divide.npy", array)
