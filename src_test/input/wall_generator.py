# wand_voxel_3d.py
# Anforderungen: numpy, matplotlib (TkAgg/QtAgg/…)
import numpy as np
import matplotlib
try:
    matplotlib.use("TkAgg")
except Exception:
    pass

import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Button, RadioButtons
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from tkinter import Tk, filedialog

# ----------------- Parsing & Helpers -----------------
def parse_point(s):
    """
    Erwartet '1,1,1' oder '1 1 1' oder '1;1;1' etc.
    Gibt int-Tupel (x, y, z) zurück (auf ganze Zellen gerundet).
    """
    for sep in [',', ';', '/']:
        s = s.replace(sep, ' ')
    parts = [p for p in s.strip().split() if p]
    if len(parts) != 3:
        raise ValueError("Bitte genau drei Zahlen angeben, z.B. 1,1,1")
    xyz = tuple(int(round(float(v))) for v in parts)
    return xyz

def parse_xyz_line(line):
    """
    Erwartet eine Export-Zeile mit 'x y z' (Leerzeichen-getrennt).
    Erlaubt auch Kommas/Semikolons/Schrägstriche als Trenner.
    Leere Zeilen und '#' Kommentare werden ignoriert, dann None.
    """
    raw = line.strip()
    if not raw or raw.startswith("#"):
        return None
    for sep in [',', ';', '/']:
        raw = raw.replace(sep, ' ')
    parts = [p for p in raw.split() if p]
    if len(parts) != 3:
        raise ValueError("Ungültige Zeile (erwarte 3 Werte): " + line.rstrip())
    return (int(parts[0]), int(parts[1]), int(parts[2]))

def bbox_inclusive(a, b):
    """liefert sortierte inkl. Grenzen für jede Achse"""
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    return lo, hi

def quad_vertices_on_plane(const_axis, const_val, lo, hi):
    """
    Erzeugt die 4 Eckpunkte eines axisausgerichteten Rechtecks.
    const_axis in {0:x,1:y,2:z}, const_val = fixer Wert,
    lo/hi: jeweils (x,y,z) Arrays mit min/max.
    """
    x0, x1 = lo[0], hi[0]
    y0, y1 = lo[1], hi[1]
    z0, z1 = lo[2], hi[2]
    if const_axis == 0:
        # Ebene x = const_val -> Rechteck in (y,z)
        return [
            (const_val, y0, z0),
            (const_val, y1, z0),
            (const_val, y1, z1),
            (const_val, y0, z1),
        ]
    elif const_axis == 1:
        # Ebene y = const_val -> Rechteck in (x,z)
        return [
            (x0, const_val, z0),
            (x1, const_val, z0),
            (x1, const_val, z1),
            (x0, const_val, z1),
        ]
    else:
        # Ebene z = const_val -> Rechteck in (x,y)
        return [
            (x0, y0, const_val),
            (x1, y0, const_val),
            (x1, y1, const_val),
            (x0, y1, const_val),
        ]

def cuboid_faces(lo, hi):
    """6 Rechtecke als Listen von 4 Punkten für die Quader-Hülle"""
    x0, x1 = lo[0], hi[0]
    y0, y1 = lo[1], hi[1]
    z0, z1 = lo[2], hi[2]
    return [
        [(x0,y0,z0),(x1,y0,z0),(x1,y1,z0),(x0,y1,z0)],  # bottom (z0)
        [(x0,y0,z1),(x1,y0,z1),(x1,y1,z1),(x0,y1,z1)],  # top (z1)
        [(x0,y0,z0),(x1,y0,z0),(x1,y0,z1),(x0,y0,z1)],  # y0
        [(x0,y1,z0),(x1,y1,z0),(x1,y1,z1),(x0,y1,z1)],  # y1
        [(x0,y0,z0),(x0,y1,z0),(x0,y1,z1),(x0,y0,z1)],  # x0
        [(x1,y0,z0),(x1,y1,z0),(x1,y1,z1),(x1,y0,z1)],  # x1
    ]

# ----------------- Daten -----------------
# Alle belegten Zellen (x,y,z) als Set, damit Export duplikatfrei ist
occupied = set()

# Aktionen-Stack für Undo: Liste von Dicts mit 'cells' (Set) und 'artists' (List[Artist])
history = []

# ----------------- Plot-Setup -----------------
plt.close('all')
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.set_title("3D Wand-/Block-Editor (Gitterzellen belegen)")
ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
ax.grid(True)
ax.set_box_aspect((1,1,1))
ax.set_xlim(0, 5); ax.set_ylim(0, 5); ax.set_zlim(0, 5)

plt.subplots_adjust(bottom=0.30)

# ----------------- Widgets -----------------
ax_p1 = plt.axes([0.08, 0.20, 0.36, 0.06])
tb_p1 = TextBox(ax_p1, "P1 (x,y,z): ", initial="1,1,1")

ax_p2 = plt.axes([0.08, 0.12, 0.36, 0.06])
tb_p2 = TextBox(ax_p2, "P2 (x,y,z): ", initial="3,3,3")

ax_mode = plt.axes([0.08, 0.04, 0.36, 0.06])
rb_mode  = RadioButtons(ax_mode, ("Wand", "Volumen"), active=0)

ax_btn_add  = plt.axes([0.47, 0.20, 0.12, 0.065]); btn_add  = Button(ax_btn_add,  "Hinzufügen")
ax_btn_undo = plt.axes([0.47, 0.12, 0.12, 0.065]); btn_undo = Button(ax_btn_undo, "Rückgängig")
ax_btn_exp  = plt.axes([0.47, 0.04, 0.12, 0.065]); btn_exp  = Button(ax_btn_exp,  "Exportieren")

# Neuer Import-Button (rechts oben über dem Status)
ax_btn_imp  = plt.axes([0.61, 0.20, 0.12, 0.065]); btn_imp  = Button(ax_btn_imp,  "Importieren")

ax_status = plt.axes([0.62, 0.04, 0.32, 0.22]); ax_status.axis("off")
status_text = ax_status.text(0, 0.98, "Bereit.", va='top', fontsize=9)

def set_status(msg):
    status_text.set_text(str(msg))
    fig.canvas.draw_idle()

# ----------------- Mal-Helfer -----------------
def add_poly(vertices, facecolor=(0,0,1,0.25), edgecolor=(0,0,0,0.4), linewidth=1.2):
    poly = Poly3DCollection([vertices], facecolors=[facecolor], edgecolors=[edgecolor], linewidths=[linewidth])
    ax.add_collection3d(poly)
    return poly

def add_cuboid_wire(lo, hi, facecolor=(0,0,1,0.12), edgecolor=(0,0,0,0.35), linewidth=1.0):
    faces = cuboid_faces(lo, hi)
    poly = Poly3DCollection(faces, facecolors=[facecolor]*6, edgecolors=[edgecolor]*6, linewidths=[linewidth]*6)
    ax.add_collection3d(poly)
    return poly

def add_scatter(points_xyz, size=10, alpha=0.65):
    """
    Zeichnet eine 3D-Punktwolke für gegebene (N,3)-Koordinaten.
    Gibt den Artist zurück (für Undo).
    """
    pts = np.asarray(points_xyz, dtype=float)
    sc = ax.scatter(pts[:,0], pts[:,1], pts[:,2], s=size, alpha=alpha, depthshade=True)
    return sc

def autoscale_to_occupied():
    if occupied:
        pts = np.array(list(occupied))
        pad = np.maximum(1, (pts.max(axis=0) - pts.min(axis=0)).max() // 10 + 1)
        lo = pts.min(axis=0) - pad
        hi = pts.max(axis=0) + pad
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    else:
        ax.set_xlim(0, 5); ax.set_ylim(0, 5); ax.set_zlim(0, 5)

# ----------------- Kernlogik -----------------
def add_wall_or_volume(p1, p2, mode):
    """
    p1/p2: int-Tupel
    mode: "Wand" oder "Volumen"
    Rückgabe: (cells_added_set, artists_list, info_str)
    """
    a = np.array(p1, dtype=int)
    b = np.array(p2, dtype=int)
    lo, hi = bbox_inclusive(a, b)

    same = (a == b)
    n_same = int(same.sum())

    cells = set()
    artists = []

    if tuple(lo) == tuple(hi):
        # Ein einzelnes Feld
        cells.add(tuple(lo))
        # Visual: kleines Quadrat auf jeder Seite? Wir zeichnen eine zarte Hülle (Punkt wäre zu klein).
        artists.append(add_cuboid_wire(lo, hi, facecolor=(0,0,1,0.18)))
        return cells, artists, f"1 Zelle belegt bei {tuple(lo)}"

    if mode == "Wand" and n_same == 1:
        # Genau eine Achse ist gleich -> Rechteck-Wand
        const_axis = int(np.where(same)[0][0])
        const_val  = int(a[const_axis])
        # Zellen: alle Integer in den beiden variablen Achsen inkl. Grenzen
        X = range(int(lo[0]), int(hi[0])+1)
        Y = range(int(lo[1]), int(hi[1])+1)
        Z = range(int(lo[2]), int(hi[2])+1)
        if const_axis == 0:
            for y in Y:
                for z in Z:
                    cells.add((const_val, y, z))
        elif const_axis == 1:
            for x in X:
                for z in Z:
                    cells.add((x, const_val, z))
        else:
            for x in X:
                for y in Y:
                    cells.add((x, y, const_val))
        # Visual: das Rechteck zeichnen
        verts = quad_vertices_on_plane(const_axis, const_val, lo, hi)
        artists.append(add_poly(verts))
        return cells, artists, f"Wand in Achse {['X','Y','Z'][const_axis]}={const_val} mit {len(cells)} Zellen"

    # Volumen (explizit gewählt oder automatisch, wenn es keine reine Wand sein kann)
    X = range(int(lo[0]), int(hi[0])+1)
    Y = range(int(lo[1]), int(hi[1])+1)
    Z = range(int(lo[2]), int(hi[2])+1)
    for x in X:
        for y in Y:
            for z in Z:
                cells.add((x, y, z))
    artists.append(add_cuboid_wire(lo, hi))
    if mode == "Wand" and n_same != 1:
        info = f"Eingabe passt nicht zu einer Wand (gleich gesetzte Koordinaten: {n_same}). Volumen belegt: {len(cells)} Zellen"
    else:
        info = f"Volumen belegt: {len(cells)} Zellen"
    return cells, artists, info

# ----------------- Import-Logik -----------------
def import_from_file(fname):
    """
    Liest eine Export-Datei (Zeilen 'x y z') ein, fügt alle Zellen hinzu
    und erzeugt eine Visualisierung (Scatter + Hüllquader).
    Rückgabe: (cells_added_set, artists_list, info_str)
    """
    new_cells = set()
    bad_lines = 0
    with open(fname, "r", encoding="utf-8") as f:
        for line in f:
            try:
                xyz = parse_xyz_line(line)
                if xyz is None:
                    continue
                new_cells.add(tuple(xyz))
            except Exception:
                bad_lines += 1

    if not new_cells:
        return set(), [], ("Keine gültigen Zellen gefunden." if bad_lines == 0
                           else f"Keine gültigen Zellen; fehlerhafte Zeilen: {bad_lines}")

    # Visualisierung der importierten Daten
    artists = []
    pts = np.array(list(new_cells))
    # Punktwolke
    artists.append(add_scatter(pts, size=10, alpha=0.65))
    # Hüllquader über den importierten Bereich
    # lo = pts.min(axis=0)
    # hi = pts.max(axis=0)
    # artists.append(add_cuboid_wire(lo, hi, facecolor=(0,0,1,0.10)))

    info = f"Importiert: {len(new_cells)} Zellen"
    if bad_lines:
        info += f" (übersprungene Zeilen: {bad_lines})"
    return new_cells, artists, info

# ----------------- Callbacks -----------------
def on_add(event):
    try:
        p1 = parse_point(tb_p1.text)
        p2 = parse_point(tb_p2.text)
        mode = rb_mode.value_selected  # "Wand" oder "Volumen"

        cells, artists, info = add_wall_or_volume(p1, p2, mode)

        # Global belegen & History-Eintrag
        for c in cells: occupied.add(c)
        history.append({"cells": cells, "artists": artists})

        autoscale_to_occupied()
        fig.canvas.draw_idle()
        set_status(f"Hinzugefügt: {info}")
    except Exception as e:
        set_status(f"Fehler: {e}")

def on_undo(event):
    if not history:
        set_status("Nichts zum Rückgängig machen.")
        return
    last = history.pop()
    # belegte Zellen entfernen
    for c in last["cells"]:
        if c in occupied:
            occupied.remove(c)
    # Artists entfernen
    for art in last["artists"]:
        try:
            art.remove()
        except Exception:
            pass
    autoscale_to_occupied()
    fig.canvas.draw_idle()
    set_status("Letzte Aktion rückgängig gemacht.")

def on_export(event):
    if not occupied:
        set_status("Es gibt nichts zu exportieren.")
        return
    # Dateidialog
    try:
        root = Tk(); root.withdraw()
        fname = filedialog.asksaveasfilename(
            title="Export speichern",
            defaultextension=".txt",
            filetypes=[("Textdatei", "*.txt"), ("Alle Dateien", "*.*")]
        )
        root.destroy()
    except Exception:
        fname = "export.txt"
    if not fname:
        set_status("Export abgebrochen.")
        return
    try:
        with open(fname, "w", encoding="utf-8") as f:
            for x, y, z in sorted(occupied):
                f.write(f"{x} {y} {z}\n")
        set_status(f"Exportiert: {fname} ({len(occupied)} Zeilen)")
    except Exception as e:
        set_status(f"Fehler beim Speichern: {e}")

def on_import(event):
    # Datei wählen
    try:
        root = Tk(); root.withdraw()
        fname = filedialog.askopenfilename(
            title="Export-Datei importieren",
            filetypes=[("Textdatei", "*.txt"), ("Alle Dateien", "*.*")]
        )
        root.destroy()
    except Exception:
        fname = None
    if not fname:
        set_status("Import abgebrochen.")
        return

    try:
        cells, artists, info = import_from_file(fname)

        if not cells:
            # Nichts gültiges gefunden
            for art in artists:
                try: art.remove()
                except Exception: pass
            set_status(info)
            return

        # Globale Belegung updaten und History-Eintrag anhängen
        for c in cells: occupied.add(c)
        history.append({"cells": cells, "artists": artists})

        autoscale_to_occupied()
        fig.canvas.draw_idle()
        set_status(f"{info}: {fname}")
    except Exception as e:
        set_status(f"Fehler beim Import: {e}")

btn_add.on_clicked(on_add)
btn_undo.on_clicked(on_undo)
btn_exp.on_clicked(on_export)
btn_imp.on_clicked(on_import)

set_status("Modus wählen (Wand/Volumen) → P1/P2 eingeben → Hinzufügen. Export/Import schreiben/lesen alle belegten Zellen (x y z).")
plt.show()
