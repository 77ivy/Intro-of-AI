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
                graph[start_node].append((end_node, dista))  # (neighbor, distance)
                distance[(start_node, end_node)] = dista

    #path = list()
    dist = float(0.0)
    num_visited = int(0)

    pq = PriorityQueue()
    pq.put((0, [start]))  # Initialize the priority queue with a cost of 0 for the initial node.

    """
    visited_node: curr's last node
    visit: to avoid iterate a same node twice
    """
    #visited_node = {}
    visit = set()
    visit.add(start)

    ## deploy UCS
    while not pq.empty():
        curr_cost, path = pq.get()
        curr = path[-1] # Get the last node
        visit.add(curr)
        if curr == end:
            dist = findDist(path, distance)
            return path, dist, num_visited
        
        for neighbor, nei_cost in graph.get(curr, []):
            if neighbor not in visit:
                new_path = list(path)
                new_path.append(neighbor)
                num_visited += 1
                pq.put((curr_cost + nei_cost, new_path))  # Consider the total cost when placing it into the queue.

    return path, dist, num_visited

if __name__ == '__main__':
    path, dist, num_visited = ucs(1718165260, 8513026827)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')
