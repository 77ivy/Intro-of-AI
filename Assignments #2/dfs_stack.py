import csv
from collections import deque

edgeFile = 'edges.csv'

def findDist(path, distance):
    """
    3. to calculate the dist
    """
    dis = 0
    for i in range(1, len(path)):
        dis += distance[(path[i-1], path[i])]
    return dis

def dfs(start, end):

    """
    1. load the file
        *graph: store start node and end node(adjacent node, which means that may not only one node)
        *distance: store distance two node's distance
    """
    graph = {}
    distance = {}

    with open(edgeFile) as row:
        for r in row:
            info = r.strip().split(",")
            if info[0] != "start":
                start_node, end_node, dista, lim_speed = info
                start_node = int(start_node)
                end_node = int(end_node)
                dista = float(dista)
                if start_node not in graph:
                    graph[start_node] = []
                graph[start_node].append(end_node)
                distance[(start_node, end_node)] = dista

    path = list()
    dist = float(0.0)
    num_visited = int(0)

    stack = deque([start]) 

    """
    visited_node: curr's last node
    visit: to avoid iterate a same node twice
    """
    visited_node = {}
    visit = set()
    visit.add(start)

    ## deploy DFS
    while stack:
        curr = stack.pop()

        if curr == end:

            """
            line 61-63: get path
            """
            while curr is not None:
                path.append(curr)
                curr = visited_node.get(curr)

            path.reverse()

            dist = findDist(path, distance)

        
        for new in graph.get(curr, []):
            if new not in visit:
                visit.add(new)
                visited_node[new] = curr
                num_visited += 1
                stack.append(new)

    return path, dist, num_visited

if __name__ == '__main__':
    path, dist, num_visited = dfs(2270143902, 1079387396)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')
