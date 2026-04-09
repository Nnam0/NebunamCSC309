# ==========================
# Drone Delivery Navigation
# ==========================

import heapq
from collections import deque

# --------- Graph Definition ---------
graph = {
    'A': {'B': 2, 'C': 5, 'D': 1},
    'B': {'A': 2, 'D': 2, 'E': 3},
    'C': {'A': 5, 'D': 2, 'F': 3},
    'D': {'A': 1, 'B': 2, 'C': 2, 'E': 1, 'F': 4},
    'E': {'B': 3, 'D': 1, 'G': 2},
    'F': {'C': 3, 'D': 4, 'G': 1},
    'G': {'E': 2, 'F': 1, 'H': 3},
    'H': {'G': 3}
}

heuristic = {
    'A': 7, 'B': 6, 'C': 6, 'D': 4, 'E': 2, 'F': 2, 'G': 1, 'H': 0
}

start = 'A'
goal = 'H'

# ===================================
# Depth-First Search (DFS)
# ===================================
def dfs(graph, start, goal):
    visited = set()
    
    def dfs_helper(node, current_path):
        if node == goal:
            return current_path
        
        visited.add(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                result = dfs_helper(neighbor, current_path + [neighbor])
                if result: return result
        return None

    return dfs_helper(start, [start])

# ===================================
# Breadth-First Search (BFS)
# ===================================
def bfs(graph, start, goal):
    visited = {start}
    queue = deque([(start, [start])])

    while queue:
        (node, path) = queue.popleft()
        if node == goal:
            return path
        
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    return None

# ===================================
# Uniform Cost Search (UCS)
# ===================================
def ucs(graph, start, goal):
    frontier = []
    # (cumulative_cost, current_node, path)
    heapq.heappush(frontier, (0, start, [start]))
    explored = {} # Using a dict to store the lowest cost to reach a node

    while frontier:
        cost, node, path = heapq.heappop(frontier)
        
        if node == goal:
            return path
        
        if node not in explored or cost < explored[node]:
            explored[node] = cost
            for neighbor, weight in graph[node].items():
                heapq.heappush(frontier, (cost + weight, neighbor, path + [neighbor]))
    return None

# ===================================
# A* Search
# ===================================
def a_star(graph, start, goal, heuristic):
    frontier = []
    # (priority f(n), actual_cost g(n), current_node, path)
    heapq.heappush(frontier, (heuristic[start], 0, start, [start]))
    explored = {}

    while frontier:
        f, g, node, path = heapq.heappop(frontier)
        
        if node == goal:
            return path
        
        if node not in explored or g < explored[node]:
            explored[node] = g
            for neighbor, weight in graph[node].items():
                new_g = g + weight
                new_f = new_g + heuristic[neighbor]
                heapq.heappush(frontier, (new_f, new_g, neighbor, path + [neighbor]))
    return None

# ===================================
# Run and Compare
# ===================================
if __name__ == "__main__":
    print("DFS Path:", dfs(graph, start, goal))
    print("BFS Path:", bfs(graph, start, goal))
    print("UCS Path:", ucs(graph, start, goal))
    print("A* Path :", a_star(graph, start, goal, heuristic))