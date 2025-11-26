import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Parameters
R = 4          # number of regions
S = 8          # number of segments
d = 3          # degree -> d+1 control points per segment

fig, ax = plt.subplots(figsize=(6, 3))

# Draw corridor regions as vertical strips
for r in range(R):
    ax.add_patch(Rectangle((r, -1), 1, 2,
                           fill=False, linewidth=1))
    ax.text(r + 0.5, 1.05, f"R{r+1}", ha="center", va="bottom")

# Simple nondecreasing assignment r(s)
# (e.g., every 2 segments we move to the next region)
segment_regions = np.minimum(S // (R-1), np.arange(S) // 2)
segment_regions = np.clip(segment_regions, 0, R-1)

# Generate and plot control points for each segment
for s in range(S):
    r = segment_regions[s]
    x_center = r + 0.5

    # (d+1) control points around the center, some jitter in y
    xs = x_center + 0.15 * (np.linspace(-1, 1, d+1))
    ys = 0.2 * np.random.randn(d+1)

    # mark control points
    ax.plot(xs, ys, "-o", label=None, alpha=0.8)

# Example "violating" control point outside any region
p_viol = np.array([-0.2, 0.6])
ax.plot(p_viol[0], p_viol[1], "rx", markersize=8, label="violation")
ax.text(p_viol[0], p_viol[1]+0.1, "half-space\nviolation", ha="center")

ax.set_xlim(-0.5, R + 0.5)
ax.set_ylim(-1.2, 1.4)
ax.set_xlabel("Region index (nondecreasing)")
ax.set_ylabel("y")
ax.set_aspect("equal", adjustable="box")
ax.set_title("Segments, corridor regions, and control points")
plt.tight_layout()
plt.show()
