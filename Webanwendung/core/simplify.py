import numpy as np

def simplify_rdp(path, epsilon=1.0):
    pts = np.array([[c,r] for (r,c) in path], dtype=float)
    def rdp(points, eps):
        if len(points)<3: return points
        start,end=points[0],points[-1]
        line=end-start; L=np.linalg.norm(line)
        if L==0: return np.vstack([start,end])
        d=np.abs(np.cross(line, points[1:-1]-start))/L
        idx=np.argmax(d)
        if d[idx]>eps:
            left=rdp(points[:idx+2],eps)
            right=rdp(points[idx+1:],eps)
            return np.vstack([left[:-1],right])
        else: return np.vstack([start,end])
    simp=rdp(pts,epsilon).astype(int)
    return [(int(r),int(c)) for (c,r) in simp]
