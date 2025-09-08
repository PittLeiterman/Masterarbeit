import math, heapq

def astar(rows, cols, blocks, start, goal):
    def in_bounds(rc): r,c=rc; return 0<=r<rows and 0<=c<cols
    directions=[(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
    def neighbors(rc):
        r,c=rc
        for dr,dc in directions:
            nxt=(r+dr,c+dc)
            if not in_bounds(nxt) or nxt in blocks: continue
            if dr and dc:
                if (r+dr,c) in blocks or (r,c+dc) in blocks: continue
            yield nxt
    def cost(a,b): return 1.0 if a[0]==b[0] or a[1]==b[1] else math.sqrt(2)
    def heuristic(a,b):
        dx,dy=abs(a[1]-b[1]),abs(a[0]-b[0])
        return (max(dx,dy)-min(dx,dy))+min(dx,dy)*math.sqrt(2)
    g={start:0}; open_heap=[(heuristic(start,goal),0,start)]; came={}; closed=set(); push=1
    while open_heap:
        _,_,cur=heapq.heappop(open_heap)
        if cur in closed: continue
        if cur==goal:
            path=[cur]
            while cur in came: cur=came[cur]; path.append(cur)
            return list(reversed(path))
        closed.add(cur)
        for nxt in neighbors(cur):
            t=g[cur]+cost(cur,nxt)
            if nxt in g and t>=g[nxt]: continue
            came[nxt]=cur; g[nxt]=t
            heapq.heappush(open_heap,(t+heuristic(nxt,goal),push,nxt)); push+=1
    return None
