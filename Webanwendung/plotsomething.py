import numpy as np
import matplotlib.pyplot as plt

def bezier_cubic(P, t):
    """Evaluate a cubic Bézier curve with control points P (4x2) at parameters t."""
    P = np.asarray(P, dtype=float)
    t = t[:, None]  # (N, 1)
    B0 = (1 - t) ** 3
    B1 = 3 * (1 - t) ** 2 * t
    B2 = 3 * (1 - t) * t ** 2
    B3 = t ** 3
    return B0 * P[0] + B1 * P[1] + B2 * P[2] + B3 * P[3]

# --- First segment: choose some nice shape ---
P0 = np.array([
    [0.0, 0.0],
    [1.0, 1.2],
    [2.0, 1.0],
    [3.0, 0.3],
])

segments = [P0]

# --- Build next segment with C² continuity at the junction ---
def next_c2_segment(prev_seg, new_end):
    """
    Given previous cubic Bézier prev_seg with control points [P0,P1,P2,P3]
    and a desired end point for the new segment, construct [Q0,Q1,Q2,Q3]
    such that:
        Q0 = P3    (C0)
        Q1 = 2*P3 - P2      (C1)
        Q2 = P1 + 4*P3 - 4*P2   (C2)
        Q3 = new_end   (free to choose)
    """
    A, B, C = prev_seg[1], prev_seg[2], prev_seg[3]
    Q0 = C
    Q1 = 2 * C - B           # C1 continuity
    Q2 = A + 4 * C - 4 * B   # C2 continuity
    Q3 = np.asarray(new_end, dtype=float)
    return np.vstack([Q0, Q1, Q2, Q3])

# Anchor/end points for each segment
anchors = [
    segments[0][0],       # start (unused after first)
    segments[0][3],       # end of first
    [7.5, 0.0],
    [13.0, 0.8],
    [17.0, 0.2],
]

# Build 3 more segments with C² continuity
segments.append(next_c2_segment(segments[0], anchors[2]))
segments.append(next_c2_segment(segments[1], anchors[3]))
segments.append(next_c2_segment(segments[2], anchors[4]))

# --- Plotting ---
t = np.linspace(0.0, 1.0, 200)
fig, ax = plt.subplots(figsize=(8, 3))

# Blanc (white) background, no grid, no legend
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# Plot all segments
for P in segments:
    C = bezier_cubic(P, t)
    ax.plot(C[:, 0], C[:, 1], linewidth=2)

# Highlight control points + Bézier hull of one segment (e.g. second)
idx = 1
P_h = segments[idx]

ax.scatter(P_h[:, 0], P_h[:, 1], s=50, c='red', zorder=5)
ax.plot(P_h[:, 0], P_h[:, 1], '--', c='red', linewidth=1.5)
ax.fill(P_h[:, 0], P_h[:, 1], alpha=0.2, color='red')

# Make it clean: no axes, ticks or frame
ax.set_aspect('equal', 'box')
ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)

plt.tight_layout()
plt.show()
