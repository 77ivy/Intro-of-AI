import csv
from queue import PriorityQueue

edgeFile = 'edges.csv'

def findDist(path, distance):
    """
    3. to calculate the dist
    """
    dis = 0
    for i in range(1, len(path)):
        dis += distance[(path[i-1], path[i])]
    return dis

def ucs(start, end):

    """
    1. load the file
        *graph: store start node and end node(adjacent node, which means that may not only one node)
        *distance: store distance two node's distance
    """
    graph = {}
    distance = {}

    with open(edgeFile) as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header row
        for row in reader:
            start_node, end_node, dista, lim_speed = map(int, row)
            dista = float(dista)
            if start_node not in graph:
                graph[start_node] = []
            graph[start_node].append((end_node, dista))  # 存储为 (邻居节点, 距离) 元组
            distance[(start_node, end_node)] = dista

    dist = float(0.0)
    num_visited = int(0)

    pq = PriorityQueue()
    pq.put((0, [start]))  # 初始化优先队列，初始节点的代价为0

    visited = set()  # 存储已经访问过的节点

    while not pq.empty():
        curr_cost, path = pq.get()
        curr = path[-1]
        visited.add(curr)
        if curr == end:
            dist = findDist(path, distance)
            return path, dist, num_visited
        
        for neighbor, neighbor_cost in graph.get(curr, []):
            if neighbor not in visited:
                new_path = list(path)
                new_path.append(neighbor)
                num_visited += 1
                pq.put((curr_cost + neighbor_cost, new_path))  # 放入队列时考虑总代价

    return None, 0, num_visited

if __name__ == '__main__':
    path, dist, num_visited = ucs(426882161, 1737223506)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')
