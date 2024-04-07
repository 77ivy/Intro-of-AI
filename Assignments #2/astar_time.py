import csv
from queue import PriorityQueue

edgeFile = 'edges.csv'
heuristicFile = 'heuristic.csv'

def find_time(path, time):
    """
    Function to calculate the total time of a given path.
    """
    total_time = 0
    for i in range(1, len(path)):
        total_time += time[(path[i-1], path[i])]
    return total_time

def astar_time(start, end):
    """
    A* Search Algorithm to find the fastest path.
    """
    # Initialize graph and time dictionaries
    graph = {}
    time = {}

    # Load edge information from edges.csv
    with open(edgeFile) as row:
        for r in row:
            info = r.strip().split(",")
            if info[0] != "start":
                start_node, end_node, dist, speed_limit = info
                start_node = int(start_node)
                end_node = int(end_node)
                dist = float(dist)
                speed_limit = float(speed_limit)
                if start_node not in graph:
                    graph[start_node] = {}
                graph[start_node][end_node] = dist / speed_limit  # time = distance / speed
                time[(start_node, end_node)] = dist / speed_limit

    # Load heuristic information from heuristic.csv
    heuristic = {}
    with open(heuristicFile) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            heuristic[int(row['node'])] = float(row[str(end)])

    # Initialize priority queue, visited set, and parent dictionary
    pq = PriorityQueue()
    pq.put((0, start))
    visit = set()
    parent = {}
    g_val = {start: 0}

    while not pq.empty():
        _, curr = pq.get()

        if curr == end:
            path = [curr]
            while curr != start:
                curr = parent[curr]
                path.append(curr)
            path.reverse()
            total_time = find_time(path, time)
            return path, total_time, len(visit)

        visit.add(curr)
        for neighbor, dist in graph.get(curr, {}).items():
            if neighbor in visit:
                continue
            tg = g_val[curr] + dist
            if neighbor not in g_val or tg < g_val[neighbor]:
                g_val[neighbor] = tg
                f_val = tg + heuristic.get(neighbor, float('inf'))
                pq.put((f_val, neighbor))
                parent[neighbor] = curr

    return None, 0, len(visit)

if __name__ == '__main__':
    path, time, num_visited = astar_time(1718165260, 8513026827)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total second of path: {time}')
    print(f'The number of visited nodes: {num_visited}')
