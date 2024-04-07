import csv
edgeFile = 'edges.csv'
heuristicFile = 'heuristic.csv'

def readFile(end):
    """Read edges.csv, save the adjacency list of the graph, and save distances between two nodes.
    Read heuristic.csv, save the heuristic function of each node.

    Args:
        end (int): the end node ID

    Returns:
        graph: a dictionary, where the key is the node and value is a list of adjacent nodes.
        distances: a dictionary, where the key is a tuple with two adjacent nodes and value is the distance between them.
        heuristic: a dictionary, where the key is the node and the value is the corresponding value of the heuristic function of the node.
    """
    graph = {}
    distances = {}
    heuristic = {}
    with open(edgeFile, newline='') as csvfile: # read edgeFile and save the graph and the distances
        rows = csv.DictReader(csvfile)
        for row in rows:
            if int(row['start']) not in graph:
                graph[int(row['start'])]=[]
            graph[int(row['start'])].append(int(row['end']))
            distances[(int(row['start']), int(row['end']))] = float(row['distance'])
            
    with open(heuristicFile, newline='') as csvfile: # read heuristicFile and save the heuristic funtion
        rows = csv.DictReader(csvfile)
        for row in rows:
            heuristic[int(row['node'])] = float(row[str(end)])
    return graph, distances, heuristic

def rebuildPath(parent, start, end):
    """Rebuild the path from start node to end node with the saved parent relationship.

    Args:
        parent (dictionary): saves parents of the node
        start (int): start node ID
        end (int): end node ID

    Returns:
        path: a list with the start node to end node.
    """
    path = [end]
    while path[-1] != start:
        path.append(parent[path[-1]])
    path.reverse()
    return path

def computeDistance(path, distances):
    """Compute the total distance of the path.

    Args:
        path (list): saves the nodes on the path
        distances (dictionary): save the distances between two nodes

    Returns:
        distance: int, the total distance along the path 
    """
    distance = 0
    for i in range(len(path) - 1):
        distance += distances[(path[i], path[i + 1])]
    return distance

def astar(start, end):
    """Use A* search to find the optimal path from start node to end node

    Args:
        start (int): start node ID
        end (int): end node ID

    Returns:
        path: the path from start node to end node
        dist: the total distance of the path
        num_visited: the number of nodes visited by A* search
    """
    graph, distances, heuristic = readFile(end)
    open_list = [start] # use a open list to implement A* search
    visited = [start]
    g = {} # use dictionary to store the current distances from start to all other nodes
    g[start] = 0 
    parent = {} # use dictionary to store the parent of a node
    while len(open_list) > 0:
        current_node = None
        
        for v in open_list: # find a node with the lowest value of f()
            if current_node == None or g[v] + heuristic[v] < g[current_node] + heuristic[current_node]:
                current_node = v
        
        if current_node == end: # if the current node is the end node
            path = rebuildPath(parent, start, end) # reconstruct the path 
            distance = computeDistance(path, distances) # calculate the distance of the path
            return path, distance, len(visited)
        
        for neighbor in graph.get(current_node, []): # iterate each neighbor node of the current node
            
            if neighbor not in open_list and neighbor not in visited: # if the current node isn't in open list and haven't visited
                open_list.append(neighbor) # add it to open list
                parent[neighbor] = current_node # record the parent of the neighbor node
                g[neighbor] = g[current_node] + distances[(current_node, neighbor)] # update the distance between start node and this neighbor node
                
            else: # if the current node is in open list or visited
                if g[neighbor] > g[current_node] + distances[(current_node, neighbor)]: # check if the distance from start node to this neighbor node is larger than from start node to the current node then go to the neighbor node
                    g[neighbor] = g[current_node] + distances[(current_node, neighbor)] # if the distance is larger, which means go to the current node then go to the neighbor is smaller, update the distance from start node to the neighbor node
                    parent[neighbor] = current_node # record the parent of the neighbor node
                    if neighbor in visited: # if the node was in the visited, move it to open list. (because we need to check it again)
                        visited.remove(neighbor)
                        open_list.append(neighbor)
        
        open_list.remove(current_node) # remove the current node from the open list
        visited.append(current_node) # add the current node to visited
    return None, 0, len(visited) # if we cannot find the route from start to end, return None.


if __name__ == '__main__':
    path, dist, num_visited = astar(2270143902, 1079387396)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')