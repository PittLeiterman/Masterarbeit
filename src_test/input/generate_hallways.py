#!/usr/bin/env python3
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import random

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

AUTO_HALL_W = 7
AUTO_HALL_T = 1
AUTO_DOOR_DEPTH = 1


def add_hollow_box(voxels, x, y, z, sx, sy, sz, t=1):
    x, y, z, sx, sy, sz, t = map(int, (x, y, z, sx, sy, sz, max(1, t)))
    if sx <= 0 or sy <= 0 or sz <= 0: return
    x2, y2, z2 = x + sx - 1, y + sy - 1, z + sz - 1
    for i in range(x, x + sx):
        dx_shell = (i - x) < t or (x2 - i) < t
        for j in range(y, y + sy):
            dy_shell = (j - y) < t or (y2 - j) < t
            for k in range(z, z + sz):
                dz_shell = (k - z) < t or (z2 - k) < t
                if dx_shell or dy_shell or dz_shell:
                    voxels.add((i, j, k))

def carve_box(voxels, x, y, z, sx, sy, sz):
    x, y, z, sx, sy, sz = map(int, (x, y, z, sx, sy, sz))
    if sx <= 0 or sy <= 0 or sz <= 0: return
    for i in range(x, x + sx):
        for j in range(y, y + sy):
            for k in range(z, z + sz):
                voxels.discard((i, j, k))

def room_bounds(room):
    x, y, z = room["x"], room["y"], room["z"]
    sx, sy, sz = room["sx"], room["sy"], room["sz"]
    return (x, x+sx-1, y, y+sy-1, z, z+sz-1)

def _extend_line_points(pts, pre=1, post=1):
    if not pts or len(pts) == 1:
        return pts
    sx0 = pts[0][0] - pts[1][0]
    sy0 = pts[0][1] - pts[1][1]
    sz0 = pts[0][2] - pts[1][2]
    sx1 = pts[-1][0] - pts[-2][0]
    sy1 = pts[-1][1] - pts[-2][1]
    sz1 = pts[-1][2] - pts[-2][2]

    pre_pts  = [(pts[0][0] + (i+1)*sx0, pts[0][1] + (i+1)*sy0, pts[0][2] + (i+1)*sz0) for i in range(pre)]
    post_pts = [(pts[-1][0] + (i+1)*sx1, pts[-1][1] + (i+1)*sy1, pts[-1][2] + (i+1)*sz1) for i in range(post)]
    return pre_pts[::-1] + pts + post_pts


def _extend_into_walls(pts, roomA, roomB):
    if not pts or len(pts) == 1:
        return pts

    def interior_bounds(r):
        return room_interior_bounds(r)

    def inside_interior(p, r):
        ix0, ix1, iy0, iy1, iz0, iz1 = interior_bounds(r)
        return point_in_box(p, ix0, ix1, iy0, iy1, iz0, iz1)

    def step_towards(p_from, p_to):
        return (p_to[0] - p_from[0], p_to[1] - p_from[1], p_to[2] - p_from[2])

    out = pts[:]

    sx, sy, sz = step_towards(pts[1], pts[0])
    cur = pts[0]
    while True:
        nxt = (cur[0] + sx, cur[1] + sy, cur[2] + sz)
        if inside_interior(nxt, roomA):
            break
        out.insert(0, nxt)
        cur = nxt

    sx, sy, sz = step_towards(pts[-2], pts[-1])
    cur = pts[-1]
    while True:
        nxt = (cur[0] + sx, cur[1] + sy, cur[2] + sz)
        if inside_interior(nxt, roomB):
            break
        out.append(nxt)
        cur = nxt

    return out



def room_interior_bounds(room):
    x, y, z = room["x"], room["y"], room["z"]
    sx, sy, sz = room["sx"], room["sy"], room["sz"]
    t = int(max(1, room["wall"]))
    return (x+t, x+sx-1-t, y+t, y+sy-1-t, z+t, z+sz-1-t)

def point_in_box(p, bx0, bx1, by0, by1, bz0, bz1):
    x,y,z = p
    return (bx0 <= x <= bx1) and (by0 <= y <= by1) and (bz0 <= z <= bz1)

def point_in_any_room_interior(p, rooms):
    for r in rooms:
        ix0, ix1, iy0, iy1, iz0, iz1 = room_interior_bounds(r)
        if ix0 > ix1 or iy0 > iy1 or iz0 > iz1:
            continue
        if point_in_box(p, ix0, ix1, iy0, iy1, iz0, iz1):
            return True
    return False

def clip_line_points_outside_rooms(pts, roomA, roomB):
    a_ix0,a_ix1,a_iy0,a_iy1,a_iz0,a_iz1 = room_interior_bounds(roomA)
    b_ix0,b_ix1,b_iy0,b_iy1,b_iz0,b_iz1 = room_interior_bounds(roomB)

    n = len(pts)
    start = 0
    while start < n:
        p = pts[start]
        if not point_in_box(p, a_ix0,a_ix1,a_iy0,a_iy1,a_iz0,a_iz1):
            break
        start += 1

    end = n - 1
    while end >= start:
        p = pts[end]
        if not point_in_box(p, b_ix0,b_ix1,b_iy0,b_iy1,b_iz0,b_iz1):
            break
        end -= 1

    return pts[start:end+1]


def raster_line_3d(a, b):
    ax, ay, az = map(float, a)
    bx, by, bz = map(float, b)
    dx, dy, dz = bx - ax, by - ay, bz - az
    steps = int(max(abs(dx), abs(dy), abs(dz), 1))
    pts = []
    for s in range(steps + 1):
        t = s / steps
        x = int(round(ax + t * dx))
        y = int(round(ay + t * dy))
        z = int(round(az + t * dz))
        if not pts or (x, y, z) != pts[-1]:
            pts.append((x, y, z))
    return pts

def add_hollow_cube_centered(voxels, cx, cy, cz, w, t=1):
    w = int(w); t = int(max(1, t))
    if w <= 0: return
    if w % 2 == 0: w += 1
    half = w // 2
    add_hollow_box(voxels, cx - half, cy - half, cz - half, w, w, w, t)

def add_hollow_rect_hallway(voxels, a, b, w=3, t=1):
    for (x, y, z) in raster_line_3d(a, b):
        add_hollow_cube_centered(voxels, x, y, z, w, t)

def punch_connection_to_hall(voxels, room, towards, width=5, overreach=2):
    x, y, z = room["x"], room["y"], room["z"]
    sx, sy, sz = room["sx"], room["sy"], room["sz"]
    t = int(max(1, room["wall"]))
    cx, cy, cz = room["cx"], room["cy"], room["cz"]

    w = int(max(1, width));  w += (w % 2 == 0)
    half = w // 2

    L = 1 + t + int(max(0, overreach))

    dx, dy, dz = towards[0] - cx, towards[1] - cy, towards[2] - cz
    ax, ay, az = abs(dx), abs(dy), abs(dz)

    if ax >= ay and ax >= az:
        if dx >= 0:
            start_x = (x + sx - 1) - t
        else:
            start_x = (x + t) - L + 1
        carve_box(voxels, start_x, cy - half, cz - half, L, w, w)

    elif ay >= ax and ay >= az:
        if dy >= 0:
            start_y = (y + sy - 1) - t
        else:
            start_y = (y + t) - L + 1
        carve_box(voxels, cx - half, start_y, cz - half, w, L, w)

    else:
        if dz >= 0:
            start_z = (z + sz - 1) - t
        else:
            start_z = (z + t) - L + 1
        carve_box(voxels, cx - half, cy - half, start_z, w, w, L)



def add_hollow_rect_hallway_clipped(voxels, rooms, roomA, roomB, w=3, t=1):
    a = (roomA["cx"], roomA["cy"], roomA["cz"])
    b = (roomB["cx"], roomB["cy"], roomB["cz"])

    pts = raster_line_3d(a, b)
    pts = clip_line_points_outside_rooms(pts, roomA, roomB)
    if not pts:
        return

    pts = _extend_into_walls(pts, roomA, roomB)


    w_eff = int(max(1, w))
    if w_eff % 2 == 0:
        w_eff += 1
    t_eff = int(max(1, t))
    half = w_eff // 2

    pad = max(1, min(int(roomA["wall"]), int(roomB["wall"])))
    pts = _extend_line_points(pts, pre=pad, post=pad)

    for (x, y, z) in pts:
        for i in range(x - half, x + half + 1):
            for j in range(y - half, y + half + 1):
                for k in range(z - half, z + half + 1):
                    on_shell = (
                        (abs(i - x) >= half - (t_eff - 1)) or
                        (abs(j - y) >= half - (t_eff - 1)) or
                        (abs(k - z) >= half - (t_eff - 1))
                    )
                    if not on_shell:
                        continue
                    q = (i, j, k)
                    if point_in_any_room_interior(q, rooms):
                        continue
                    voxels.add(q)

    inner_w = max(1, w_eff - 2 * t_eff)
    inner_half = inner_w // 2
    for (x, y, z) in pts:
        carve_box(voxels, x - inner_half, y - inner_half, z - inner_half,
                  inner_w, inner_w, inner_w)


def carve_door_on_room_face(voxels, room, towards, width=3, depth=1):
    x, y, z = room["x"], room["y"], room["z"]
    sx, sy, sz = room["sx"], room["sy"], room["sz"]
    cx, cy, cz = room["cx"], room["cy"], room["cz"]

    width = int(max(1, width))
    if width % 2 == 0: width += 1
    half = width // 2
    depth = int(max(1, depth))

    dx, dy, dz = towards[0] - cx, towards[1] - cy, towards[2] - cz
    ax = abs(dx); ay = abs(dy); az = abs(dz)
    if ax >= ay and ax >= az:
        pos = dx >= 0
        face_x = x + sx - 1 if pos else x
        ox = face_x - depth + 1 if pos else face_x
        oy = cy - half
        oz = cz - half
        carve_box(voxels, ox, oy, oz, depth, width, width)
    elif ay >= ax and ay >= az:
        pos = dy >= 0
        face_y = y + sy - 1 if pos else y
        oy = face_y - depth + 1 if pos else face_y
        ox = cx - half
        oz = cz - half
        carve_box(voxels, ox, oy, oz, width, depth, width)
    else:
        pos = dz >= 0
        face_z = z + sz - 1 if pos else z
        oz = face_z - depth + 1 if pos else face_z
        ox = cx - half
        oy = cy - half
        carve_box(voxels, ox, oy, oz, width, width, depth)

def build_map_shell(nx, ny, nz, t=1):
    t = max(1, int(t))
    v = set()
    for x in range(0, min(t, nx)):
        for y in range(ny):
            for z in range(nz): v.add((x, y, z))
    for x in range(max(0, nx - t), nx):
        for y in range(ny):
            for z in range(nz): v.add((x, y, z))
    for y in range(0, min(t, ny)):
        for x in range(nx):
            for z in range(nz): v.add((x, y, z))
    for y in range(max(0, ny - t), ny):
        for x in range(nx):
            for z in range(nz): v.add((x, y, z))
    for z in range(0, min(t, nz)):
        for x in range(nx):
            for y in range(ny): v.add((x, y, z))
    for z in range(max(0, nz - t), nz):
        for x in range(nx):
            for y in range(ny): v.add((x, y, z))
    return v

class CubicHallwayApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Hollow Cubes & Small Doors/Hallways (Voxel)")
        self.geometry("1100x820")

        # Map
        self.nx = tk.IntVar(value=120)
        self.ny = tk.IntVar(value=120)
        self.nz = tk.IntVar(value=60)
        self.show_shell = tk.BooleanVar(value=False)
        self.shell_t = tk.IntVar(value=1)
        self.include_shell_export = tk.BooleanVar(value=False)

        # Rooms
        self.rx = tk.IntVar(value=10); self.ry = tk.IntVar(value=10); self.rz = tk.IntVar(value=0)
        self.rsx = tk.IntVar(value=30); self.rsy = tk.IntVar(value=30); self.rsz = tk.IntVar(value=20)
        self.rwall = tk.IntVar(value=2)

        # Door 
        self.dx = tk.IntVar(value=24); self.dy = tk.IntVar(value=10); self.dz = tk.IntVar(value=8)
        self.dsx = tk.IntVar(value=2); self.dsy = tk.IntVar(value=4); self.dsz = tk.IntVar(value=6)

        # Hallway
        self.hx1 = tk.IntVar(value=40); self.hy1 = tk.IntVar(value=24); self.hz1 = tk.IntVar(value=8)
        self.hx2 = tk.IntVar(value=60); self.hy2 = tk.IntVar(value=24); self.hz2 = tk.IntVar(value=8)
        self.hR  = tk.IntVar(value=3);  self.hT  = tk.IntVar(value=1)

        # Preview
        self.max_pts = tk.IntVar(value=60000)
        self.alpha = tk.DoubleVar(value=0.18)
        self.size_pts = tk.DoubleVar(value=5.0)

        self.voxels = set()
        self.shell_cache = set()
        self.rooms = []

        self._build_ui()
        self._init_plot()

    def _build_ui(self):
        top = ttk.Frame(self); top.pack(fill="x", padx=10, pady=10)

        # Map panel
        fm = ttk.LabelFrame(top, text="Map")
        fm.pack(side="left", padx=(0,10))
        ttk.Label(fm, text="NX").grid(row=0, column=0); ttk.Entry(fm, width=6, textvariable=self.nx).grid(row=0, column=1)
        ttk.Label(fm, text="NY").grid(row=0, column=2); ttk.Entry(fm, width=6, textvariable=self.ny).grid(row=0, column=3)
        ttk.Label(fm, text="NZ").grid(row=0, column=4); ttk.Entry(fm, width=6, textvariable=self.nz).grid(row=0, column=5)
        ttk.Button(fm, text="Apply Size", command=self._apply_size).grid(row=0, column=6, padx=8)
        ttk.Checkbutton(fm, text="Show map shell", variable=self.show_shell, command=self._update_plot).grid(row=1, column=0, columnspan=3, sticky="w")
        ttk.Checkbutton(fm, text="Include shell in export", variable=self.include_shell_export).grid(row=1, column=3, columnspan=3, sticky="w")
        ttk.Label(fm, text="Shell T").grid(row=2, column=0, sticky="e")
        ttk.Spinbox(fm, from_=1, to=20, width=5, textvariable=self.shell_t, command=self._rebuild_shell).grid(row=2, column=1, sticky="w")

        # Room panel
        fr = ttk.LabelFrame(top, text="Add Hollow Cuboid")
        fr.pack(side="left", padx=(0,10))
        r=0
        for lbl, var in [("x", self.rx), ("y", self.ry), ("z", self.rz), ("sx", self.rsx), ("sy", self.rsy), ("sz", self.rsz), ("wall", self.rwall)]:
            ttk.Label(fr, text=lbl).grid(row=r, column=0, padx=4, pady=2, sticky="e")
            ttk.Entry(fr, width=7, textvariable=var).grid(row=r, column=1, padx=2, pady=2, sticky="w"); r+=1
        ttk.Button(fr, text="Add Room", command=self._add_room).grid(row=r, column=0, columnspan=2, pady=4)

        # Door panel
        fd = ttk.LabelFrame(top, text="Carve Door (rectangular hole)")
        fd.pack(side="left", padx=(0,10))
        r=0
        for lbl, var in [("x", self.dx), ("y", self.dy), ("z", self.dz), ("sx", self.dsx), ("sy", self.dsy), ("sz", self.dsz)]:
            ttk.Label(fd, text=lbl).grid(row=r, column=0, padx=4, pady=2, sticky="e")
            ttk.Entry(fd, width=7, textvariable=var).grid(row=r, column=1, padx=2, pady=2, sticky="w"); r+=1
        ttk.Button(fd, text="Carve Door", command=self._carve_door).grid(row=r, column=0, columnspan=2, pady=4)

        # Hallway panel
        fh = ttk.LabelFrame(top, text="Add Hollow Hallway (capsule shell)")
        fh.pack(side="left")
        r=0
        for lbl, var in [("x1", self.hx1), ("y1", self.hy1), ("z1", self.hz1), ("x2", self.hx2), ("y2", self.hy2), ("z2", self.hz2),
                         ("R (outer)", self.hR), ("T (wall)", self.hT)]:
            ttk.Label(fh, text=lbl).grid(row=r, column=0, padx=4, pady=2, sticky="e")
            ttk.Entry(fh, width=8, textvariable=var).grid(row=r, column=1, padx=2, pady=2, sticky="w"); r+=1
        ttk.Button(fh, text="Add Hallway", command=self._add_hallway_capsule).grid(row=r, column=0, columnspan=2, pady=4)

        # List + actions
        mid = ttk.Frame(self); mid.pack(fill="x", padx=10, pady=(4,8))
        ttk.Button(mid, text="Clear All", command=self._clear_all).pack(side="left")
        ttk.Button(mid, text="Export .txt", command=self._export).pack(side="left", padx=6)

        # Preview controls
        pc = ttk.LabelFrame(self, text="3D Preview")
        pc.pack(fill="x", padx=10, pady=6)
        ttk.Label(pc, text="Max points").grid(row=0, column=0, sticky="e", padx=6)
        ttk.Spinbox(pc, from_=2000, to=400000, increment=2000, textvariable=self.max_pts, width=10,
                    command=self._update_plot).grid(row=0, column=1, sticky="w")
        ttk.Label(pc, text="Alpha").grid(row=0, column=2, sticky="e", padx=(18,6))
        ttk.Spinbox(pc, from_=0.05, to=1.0, increment=0.05, textvariable=self.alpha, width=6,
                    command=self._update_plot).grid(row=0, column=3, sticky="w")
        ttk.Label(pc, text="Marker size").grid(row=0, column=4, sticky="e", padx=(18,6))
        ttk.Spinbox(pc, from_=1.0, to=20.0, increment=1.0, textvariable=self.size_pts, width=6,
                    command=self._update_plot).grid(row=0, column=5, sticky="w")
        ttk.Button(pc, text="Refresh 3D", command=self._update_plot).grid(row=0, column=6, padx=10)

        # 3D area
        frm_plot = ttk.LabelFrame(self, text="3D Preview (drag/scroll)")
        frm_plot.pack(fill="both", expand=True, padx=10, pady=(0,10))
        self.frm_plot = frm_plot

    def _init_plot(self):
        self.fig = Figure(figsize=(8.0, 6.4), dpi=100)
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frm_plot)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=6, pady=(6,0))
        toolbar = NavigationToolbar2Tk(self.canvas, self.frm_plot, pack_toolbar=False)
        toolbar.update(); toolbar.pack(side="bottom", fill="x")
        self._update_plot()

    # ---------------- Actions ----------------
    def _apply_size(self):
        self.voxels.clear()
        self.shell_cache.clear()
        self.rooms.clear()
        self._update_plot()

    def _rebuild_shell(self):
        nx, ny, nz = self.nx.get(), self.ny.get(), self.nz.get()
        self.shell_cache = build_map_shell(nx, ny, nz, self.shell_t.get())
        self._update_plot()

    def _add_room(self):
        nx, ny, nz = self.nx.get(), self.ny.get(), self.nz.get()
        x,y,z = self.rx.get(), self.ry.get(), self.rz.get()
        sx,sy,sz, t = self.rsx.get(), self.rsy.get(), self.rsz.get(), self.rwall.get()
        if sx<=0 or sy<=0 or sz<=0: return
        if not (0 <= x < nx and 0 <= y < ny and 0 <= z < nz):
            messagebox.showerror("Out of bounds", "Room origin must be inside map")
            return

        add_hollow_box(self.voxels, x, y, z, sx, sy, sz, t)

        room = {
            "x":x, "y":y, "z":z,
            "sx":sx, "sy":sy, "sz":sz,
            "wall":t,
            "cx": x + sx//2,
            "cy": y + sy//2,
            "cz": z + sz//2
        }
        if self.rooms:
            prev = self.rooms[-1]
            self._connect_rooms(prev, room)

        self.rooms.append(room)
        self._update_plot()

    def _connect_rooms(self, rA, rB):
        aC = (rA["cx"], rA["cy"], rA["cz"])
        bC = (rB["cx"], rB["cy"], rB["cz"])

        add_hollow_rect_hallway_clipped(
            self.voxels, self.rooms + [rA, rB], rA, rB,
            w=AUTO_HALL_W, t=AUTO_HALL_T
        )

        carve_door_on_room_face(self.voxels, rA, towards=bC, width=AUTO_HALL_W, depth=max(1, rA["wall"]))
        carve_door_on_room_face(self.voxels, rB, towards=aC, width=AUTO_HALL_W, depth=max(1, rB["wall"]))

        punch_connection_to_hall(self.voxels, rA, towards=bC, width=AUTO_HALL_W, overreach=0)
        punch_connection_to_hall(self.voxels, rB, towards=aC, width=AUTO_HALL_W, overreach=0)



    def _carve_door(self):
        x,y,z = self.dx.get(), self.dy.get(), self.dz.get()
        sx,sy,sz = self.dsx.get(), self.dsy.get(), self.dsz.get()
        carve_box(self.voxels, x, y, z, sx, sy, sz)
        self._update_plot()

    def _add_hallway_capsule(self):
        from_pt = (self.hx1.get(), self.hy1.get(), self.hz1.get())
        to_pt   = (self.hx2.get(), self.hy2.get(), self.hz2.get())
        R, T = self.hR.get(), self.hT.get()
        if R <= 0 or T <= 0:
            messagebox.showerror("Invalid", "R and T must be positive")
            return
        add_hollow_rect_hallway(self.voxels, from_pt, to_pt, w=max(1, 2*R-1), t=T)
        self._update_plot()

    def _clear_all(self):
        self.voxels.clear()
        self.shell_cache.clear()
        self.rooms.clear()
        self._update_plot()

    def _export(self):
        export_set = set(self.voxels)
        if self.include_shell_export.get():
            if not self.shell_cache:
                self._rebuild_shell()
            export_set |= self.shell_cache
        if not export_set:
            messagebox.showwarning("Nothing to export", "Build something first.")
            return
        path = filedialog.asksaveasfilename(
            title="Save occupied voxels",
            defaultextension=".txt",
            filetypes=[("Text file", "*.txt"), ("All files", "*.*")]
        )
        if not path: return
        try:
            with open(path, "w", encoding="utf-8") as f:
                for x,y,z in sorted(export_set):
                    f.write(f"{x} {y} {z}\n")
            messagebox.showinfo("Export complete", f"Saved {len(export_set)} voxels.")
        except Exception as e:
            messagebox.showerror("Export failed", str(e))

    # ---------------- Preview ----------------
    def _draw_map_wire(self, nx, ny, nz):
        xs=[0,nx]; ys=[0,ny]; zs=[0,nz]
        corners=[(xs[i],ys[j],zs[k]) for i in (0,1) for j in (0,1) for k in (0,1)]
        edges=[(0,1),(0,2),(0,4),(3,1),(3,2),(3,7),(5,1),(5,4),(5,7),(6,2),(6,4),(6,7)]
        for a,b in edges:
            xa,ya,za=corners[a]; xb,yb,zb=corners[b]
            self.ax.plot([xa,xb],[ya,yb],[za,zb], linewidth=0.8)

    def _update_plot(self):
        self.ax.clear()
        self.ax.set_xlabel("X"); self.ax.set_ylabel("Y"); self.ax.set_zlabel("Z")
        self.ax.set_title("Rooms, Doors & Auto-Connected Rectangular Hallways")

        nx, ny, nz = self.nx.get(), self.ny.get(), self.nz.get()
        self._draw_map_wire(nx, ny, nz)

        pts = list(self.voxels)
        if self.show_shell.get():
            if not self.shell_cache: self._rebuild_shell()
            pts = pts + list(self.shell_cache)

        if pts:
            vmax = max(2000, int(self.max_pts.get()))
            if len(pts) > vmax: pts = random.sample(pts, k=vmax)
            xs=[p[0] for p in pts]; ys=[p[1] for p in pts]; zs=[p[2] for p in pts]
            self.ax.scatter(xs, ys, zs, s=float(self.size_pts.get()), alpha=max(0.05, min(1.0, float(self.alpha.get()))))
        try:
            self.ax.set_box_aspect([nx, ny, nz])
        except Exception:
            self.ax.set_xlim(0,nx); self.ax.set_ylim(0,ny); self.ax.set_zlim(0,nz)
        self.canvas.draw_idle()

if __name__ == "__main__":
    CubicHallwayApp().mainloop()
