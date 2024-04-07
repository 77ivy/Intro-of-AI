import csv
from queue import PriorityQueue

edgeFile = 'edges.csv'
heuristicFile = 'heuristic.csv'

def findDist(path, distance):
    """
    Function to calculate the total distance of a given path.
    """
    total_distance = 0
    for i in range(1, len(path)):
        total_distance += distance[(path[i-1], path[i])]
    return total_distance

def astar(start, end):
    

    # Initialize graph and distance dictionaries
    graph = {}
    distance = {}

    # Load edge info
    with open(edgeFile) as row:
        for r in row:
            info = r.strip().split(",")
            if info[0] != "start":
                start_node, end_node, dista, lim_speed = info
                start_node = int(start_node)
                end_node = int(end_node)
                dista = float(dista)
                if start_node not in graph:
                    graph[start_node] = {}
                graph[start_node][end_node] = dista
                distance[(start_node, end_node)] = dista

    # Load heuristic info
    heru_ = {}
    idx = 0
    with open(heuristicFile) as row:
        for r in row:
            info = r.strip().split(",")
            if info[0] == 'node':
                for i in range(1, 4):
                    if info[i] == str(end):
                        idx = i
                        break
                continue
            heru_[info[0]] = float(info[idx])

    """
    Initialize priority queue, visited set, and parent dictionary
    """        
    pq = PriorityQueue()  
    pq.put((0, start))  # Put the start node in the priority queue with priority 0
    visit = set()  # Initialize set to store visited nodes
    parent = {}  # Initialize dictionary to store parent-child relationships
    g_val = {start: 0}  # Initialize dictionary to store g values (distance from start node)


    while not pq.empty():
        dontUse, curr = pq.get()

        if curr == end:
            path = [curr]
            while curr != start:
                curr = parent[curr]
                path.append(curr)
            path.reverse()
            total_distance = findDist(path, distance)
            return path, total_distance, len(visit)

        visit.add(curr) # Add current node to visited set
        for neighbor, dist in graph.get(curr, {}).items():
            if neighbor in visit:
                continue
            tg = g_val[curr] + dist # Calculate tentative g value (distance from start to neighbor)
            if neighbor not in g_val or tg < g_val[neighbor]: # Check if new path to neighbor is better
                g_val[neighbor] = tg # Update g value for neighbor
                f_val = tg + heru_.get(str(neighbor), float('inf'))
                pq.put((f_val, neighbor))
                parent[neighbor] = curr # Update parent of neighbor

    return path, total_distance, len(visit)

if __name__ == '__main__':
    path, dist, num_visited = astar(1718165260, 8513026827)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')
