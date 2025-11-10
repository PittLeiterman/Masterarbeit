#!/usr/bin/env python3
import math
import random
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

# 3D plotting (embedded in Tkinter)
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

DIAMETER = 10
RADIUS = DIAMETER // 2  # 5 voxels

# --------- Geometry / Voxelization ---------
def dist2_point_to_segment_squared(px, py, pz, ax, ay, az, bx, by, bz):
    abx, aby, abz = (bx - ax), (by - ay), (bz - az)
    apx, apy, apz = (px - ax), (py - ay), (pz - az)
    ab2 = abx*abx + aby*aby + abz*abz
    if ab2 == 0.0:
        dx, dy, dz = apx, apy, apz
        return dx*dx + dy*dy + dz*dz
    t = (apx*abx + apy*aby + apz*abz) / ab2
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    qx = ax + t * abx
    qy = ay + t * aby
    qz = az + t * abz
    dx, dy, dz = (px - qx), (py - qy), (pz - qz)
    return dx*dx + dy*dy + dz*dz

def fill_sphere(center, radius, voxels):
    if radius <= 0:
        return
    cx, cy, cz = center
    r2 = radius * radius
    for x in range(cx - radius, cx + radius + 1):
        for y in range(cy - radius, cy + radius + 1):
            for z in range(cz - radius, cz + radius + 1):
                dx = x - cx
                dy = y - cy
                dz = z - cz
                if dx*dx + dy*dy + dz*dz <= r2:
                    voxels.add((x, y, z))

def fill_cylinder_along_segment(a, b, radius, voxels):
    if radius <= 0:
        return
    ax, ay, az = a
    bx, by, bz = b
    r = radius
    r2 = r * r

    xmin = min(ax, bx) - r
    xmax = max(ax, bx) + r
    ymin = min(ay, by) - r
    ymax = max(ay, by) + r
    zmin = min(az, bz) - r
    zmax = max(az, bz) + r

    for x in range(xmin, xmax + 1):
        for y in range(ymin, ymax + 1):
            for z in range(zmin, zmax + 1):
                d2 = dist2_point_to_segment_squared(
                    float(x), float(y), float(z),
                    float(ax), float(ay), float(az),
                    float(bx), float(by), float(bz)
                )
                if d2 <= r2:
                    voxels.add((x, y, z))

def build_tunnel(points, radius):
    voxels = set()
    if not points or radius <= 0:
        return voxels
    fill_sphere(points[0], radius, voxels)
    for i in range(len(points) - 1):
        a = points[i]
        b = points[i + 1]
        fill_cylinder_along_segment(a, b, radius, voxels)
    if len(points) > 1:
        fill_sphere(points[-1], radius, voxels)
    return voxels

def build_tunnel_shell(points, outer_radius, wall_thickness):
    t = max(1, int(wall_thickness))
    R = int(outer_radius)
    r_inner = max(0, R - t)
    outer = build_tunnel(points, R)
    inner = build_tunnel(points, r_inner)
    return outer.difference(inner)

# --------- UI App ---------
class TunnelApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Tunnel Voxelizer (Hollow) – Diameter 10")
        self.geometry("940x720")

        self.points = []
        self.voxels = set()

        # Preview controls
        self.max_preview_var = tk.IntVar(value=20000)    # cap how many voxels to plot
        self.alpha_var = tk.DoubleVar(value=0.20)        # transparency for tunnel voxels
        self.point_size_var = tk.DoubleVar(value=18.0)   # size for path points
        self.voxel_size_var = tk.DoubleVar(value=5.0)    # size for voxel scatter

        # Tunnel params
        self.radius_var = tk.IntVar(value=RADIUS)        # outer radius
        self.wall_var = tk.IntVar(value=1)               # wall thickness (voxels)

        self.create_widgets()
        self.init_plot()

    def create_widgets(self):
        frm_top = ttk.Frame(self)
        frm_top.pack(fill="x", padx=10, pady=10)

        # Point input
        frm_in = ttk.LabelFrame(frm_top, text="Add Path Point (x y z)")
        frm_in.pack(side="left", fill="x", padx=(0,10))

        self.x_var = tk.StringVar()
        self.y_var = tk.StringVar()
        self.z_var = tk.StringVar()

        ttk.Label(frm_in, text="X:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        ttk.Entry(frm_in, textvariable=self.x_var, width=8).grid(row=0, column=1, padx=5, pady=5)
        ttk.Label(frm_in, text="Y:").grid(row=0, column=2, padx=5, pady=5, sticky="e")
        ttk.Entry(frm_in, textvariable=self.y_var, width=8).grid(row=0, column=3, padx=5, pady=5)
        ttk.Label(frm_in, text="Z:").grid(row=0, column=4, padx=5, pady=5, sticky="e")
        ttk.Entry(frm_in, textvariable=self.z_var, width=8).grid(row=0, column=5, padx=5, pady=5)

        ttk.Button(frm_in, text="Add Point", command=self.add_point).grid(row=0, column=6, padx=10, pady=5)
        ttk.Button(frm_in, text="Remove Selected", command=self.remove_selected).grid(row=0, column=7, padx=10, pady=5)

        # Tunnel parameters
        frm_params = ttk.LabelFrame(frm_top, text="Tunnel Parameters")
        frm_params.pack(side="left", fill="x", padx=(0,10))

        ttk.Label(frm_params, text="Outer radius (R):").grid(row=0, column=0, sticky="e", padx=6, pady=4)
        ttk.Spinbox(frm_params, from_=1, to=100, textvariable=self.radius_var, width=6)\
            .grid(row=0, column=1, sticky="w", padx=4, pady=4)

        ttk.Label(frm_params, text="Wall thickness (vox):").grid(row=0, column=2, sticky="e", padx=(20,6), pady=4)
        ttk.Spinbox(frm_params, from_=1, to=100, textvariable=self.wall_var, width=6)\
            .grid(row=0, column=3, sticky="w", padx=4, pady=4)

        # ---------- Points list ----------
        frm_list = ttk.LabelFrame(self, text="Path Points (in order)")
        frm_list.pack(fill="both", expand=False, padx=10, pady=(0,10), ipady=4)
        self.listbox = tk.Listbox(frm_list, height=8, font=("Consolas", 10), exportselection=False)
        self.listbox.pack(fill="both", expand=True, padx=8, pady=8)

        # ---------- Actions + status ----------
        frm_actions = ttk.Frame(self)
        frm_actions.pack(fill="x", padx=10, pady=8)

        self.status_var = tk.StringVar(value="No tunnel built yet.")
        ttk.Button(frm_actions, text="Build Tunnel", command=self.on_build).pack(side="left", padx=5)
        ttk.Button(frm_actions, text="Export .txt", command=self.on_export).pack(side="left", padx=5)
        ttk.Button(frm_actions, text="Clear", command=self.on_clear).pack(side="left", padx=5)

        ttk.Label(frm_actions, textvariable=self.status_var).pack(side="right")

        # ---------- 3D Preview Controls ----------
        frm_ctl = ttk.LabelFrame(self, text="3D Preview Controls")
        frm_ctl.pack(fill="x", padx=10, pady=(0,8))

        ttk.Label(frm_ctl, text="Max voxels to draw:").grid(row=0, column=0, sticky="e", padx=6, pady=4)
        ttk.Spinbox(frm_ctl, from_=1000, to=200000, textvariable=self.max_preview_var, increment=1000, width=10)\
            .grid(row=0, column=1, sticky="w", padx=4, pady=4)

        ttk.Label(frm_ctl, text="Tunnel alpha:").grid(row=0, column=2, sticky="e", padx=(20,6), pady=4)
        ttk.Spinbox(frm_ctl, from_=0.05, to=1.0, increment=0.05, textvariable=self.alpha_var, width=6)\
            .grid(row=0, column=3, sticky="w", padx=4, pady=4)

        ttk.Label(frm_ctl, text="Point size:").grid(row=0, column=4, sticky="e", padx=(20,6), pady=4)
        ttk.Spinbox(frm_ctl, from_=4.0, to=60.0, increment=2.0, textvariable=self.point_size_var, width=6)\
            .grid(row=0, column=5, sticky="w", padx=4, pady=4)

        ttk.Label(frm_ctl, text="Voxel marker size:").grid(row=0, column=6, sticky="e", padx=(20,6), pady=4)
        ttk.Spinbox(frm_ctl, from_=1.0, to=20.0, increment=1.0, textvariable=self.voxel_size_var, width=6)\
            .grid(row=0, column=7, sticky="w", padx=4, pady=4)

        ttk.Button(frm_ctl, text="Refresh 3D", command=self.update_plot).grid(row=0, column=8, padx=(20,6), pady=4)

        # ---------- 3D Preview Area ----------
        frm_plot = ttk.LabelFrame(self, text="3D Preview (drag to orbit, scroll to zoom)")
        frm_plot.pack(fill="both", expand=True, padx=10, pady=(0,10))
        self.frm_plot = frm_plot  # save for canvas embed

        # ---------- Info ----------
        frm_info = ttk.LabelFrame(self, text="Details")
        frm_info.pack(fill="x", padx=10, pady=(0,10))
        info = (
            f"Hollow tunnel: outer diameter {DIAMETER} (R={RADIUS}), interior empty.\n"
            "- Wall thickness is configurable; default is 1 voxel.\n"
            "- Closed spherical end caps are shell-only (hollow inside).\n"
            "- 'Export .txt' writes occupied voxels (hull only) as 'x y z'.\n"
            "- 3D preview down-samples for performance."
        )
        tk.Label(frm_info, text=info, justify="left").pack(fill="x", padx=8, pady=8)

    # ---------- 3D Plot setup ----------
    def init_plot(self):
        self.fig = Figure(figsize=(6.5, 5.5), dpi=100)
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.set_title("Path & Hollow Tunnel (preview)")

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frm_plot)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill="both", expand=True, padx=6, pady=(6,0))

        toolbar = NavigationToolbar2Tk(self.canvas, self.frm_plot, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side="bottom", fill="x")

        self.update_plot()

    def _set_equal_aspect(self, xs, ys, zs):
        if not xs:
            return
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        zmin, zmax = min(zs), max(zs)
        max_range = max(xmax - xmin, ymax - ymin, zmax - zmin, 1.0)
        cx = (xmax + xmin) / 2.0
        cy = (ymax + ymin) / 2.0
        cz = (zmax + zmin) / 2.0
        half = max_range / 2.0
        self.ax.set_xlim(cx - half, cx + half)
        self.ax.set_ylim(cy - half, cy + half)
        self.ax.set_zlim(cz - half, cz + half)
        try:
            self.ax.set_box_aspect([1, 1, 1])
        except Exception:
            pass

    def update_plot(self):
        self.ax.clear()
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.set_title("Path & Hollow Tunnel (preview)")

        # Plot path
        if self.points:
            xs = [p[0] for p in self.points]
            ys = [p[1] for p in self.points]
            zs = [p[2] for p in self.points]
            self.ax.plot(xs, ys, zs, linewidth=1.0)
            self.ax.scatter(xs, ys, zs, s=self.point_size_var.get())
        else:
            xs = ys = zs = []

        # Plot hull voxels
        if self.voxels:
            alpha = max(0.01, min(1.0, float(self.alpha_var.get())))
            vmax = max(1000, int(self.max_preview_var.get()))
            vox_list = list(self.voxels)
            n = len(vox_list)
            vox_sample = random.sample(vox_list, k=vmax) if n > vmax else vox_list
            vx = [v[0] for v in vox_sample]
            vy = [v[1] for v in vox_sample]
            vz = [v[2] for v in vox_sample]
            self.ax.scatter(vx, vy, vz, s=self.voxel_size_var.get(), alpha=alpha)
            xs = (xs or []) + vx
            ys = (ys or []) + vy
            zs = (zs or []) + vz

        self._set_equal_aspect(xs, ys, zs)
        self.canvas.draw_idle()

    # ---------- Buttons ----------
    def add_point(self):
        try:
            x = int(self.x_var.get().strip())
            y = int(self.y_var.get().strip())
            z = int(self.z_var.get().strip())
        except ValueError:
            messagebox.showerror("Invalid input", "Please enter integer values for X, Y, and Z.")
            return
        self.points.append((x, y, z))
        self.listbox.insert(tk.END, f"{len(self.points):>3}: ({x:>6}, {y:>6}, {z:>6})")
        self.x_var.set(x); self.y_var.set(y); self.z_var.set(z)
        self.update_plot()

    def remove_selected(self):
        sel = list(self.listbox.curselection())
        if not sel:
            messagebox.showinfo("Remove", "Select one or more points to remove.")
            return
        for idx in reversed(sel):
            self.listbox.delete(idx)
            del self.points[idx]
        self.listbox.delete(0, tk.END)
        for i, (x, y, z) in enumerate(self.points, start=1):
            self.listbox.insert(tk.END, f"{i:>3}: ({x:>6}, {y:>6}, {z:>6})")
        self.update_plot()

    def on_build(self):
        if not self.points:
            messagebox.showwarning("No points", "Add at least one point before building.")
            return
        R = max(1, int(self.radius_var.get()))
        T = max(1, int(self.wall_var.get()))
        if T > R:
            T = R
            self.wall_var.set(T)
        self.voxels = build_tunnel_shell(self.points, R, T)
        inner_d = 2*(R - T)
        outer_d = 2*R
        self.status_var.set(
            f"Hollow tunnel: {len(self.voxels)} voxels | outer D={outer_d}, inner D={max(0, inner_d)} (wall {T})"
        )
        self.update_plot()

    def on_export(self):
        if not self.voxels:
            messagebox.showwarning("Nothing to export", "Build the tunnel first.")
            return
        path = filedialog.asksaveasfilename(
            title="Save occupied voxels",
            defaultextension=".txt",
            filetypes=[("Text file", "*.txt"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            vox_sorted = sorted(self.voxels)
            with open(path, "w", encoding="utf-8") as f:
                for x, y, z in vox_sorted:
                    f.write(f"{x} {y} {z}\n")
            messagebox.showinfo("Export complete", f"Saved {len(vox_sorted)} voxels to:\n{path}")
        except Exception as e:
            messagebox.showerror("Export failed", f"Could not save file:\n{e}")

    def on_clear(self):
        self.points.clear()
        self.voxels.clear()
        self.listbox.delete(0, tk.END)
        self.status_var.set("Cleared.")
        self.update_plot()

if __name__ == "__main__":
    app = TunnelApp()
    app.mainloop()
