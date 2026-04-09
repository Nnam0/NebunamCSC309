import heapq
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from collections import deque

# -------------------------------
# Maze Definition
# -------------------------------
# Example 2 (Current)
maze = [
[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
[1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
[1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1],
[1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1],
[1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,0,1],
[1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1],
[1,0,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,0,1,0,1],
[1,0,1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1,0,1],
[1,0,1,0,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,0,1,0,1,0,1],
[1,0,1,0,1,0,1,0,1,0,0,0,0,0,0,0,0,0,"G",0,0,1,0,1,0,1,0,1,0,1],
[1,0,1,0,1,0,1,0,1,0,1,1,1,1,1,1,1,1,1,1,0,1,0,1,0,1,0,1,0,1],
[1,0,1,0,1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1,0,1,0,1,0,1],
[1,0,1,0,1,0,1,0,1,1,1,1,1,0,1,1,1,1,1,1,0,1,0,1,0,1,0,1,0,1],
[1,0,1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,1],
[1,0,1,0,1,0,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1],
[1,"S",0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,1],
[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
]

ROWS = len(maze)
COLS = len(maze[0])

grid = np.zeros((ROWS, COLS))
for i in range(ROWS):
    for j in range(COLS):
        if maze[i][j] == 1:
            grid[i][j] = 1
        elif maze[i][j] == "S":
            start = (i, j)
        elif maze[i][j] == "G":
            goal = (i, j)

# -------------------------------
# Node Class
# -------------------------------
class Node:
    def __init__(self, state, parent=None, action=None, cost=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.cost = cost 

# -------------------------------
# Problem Definition
# -------------------------------
def actions(state):
    x, y = state
    moves = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}
    possible = []
    for action, (dx, dy) in moves.items():
        nx, ny = x + dx, y + dy
        if 0 <= nx < ROWS and 0 <= ny < COLS:
            if maze[nx][ny] != 1:
                possible.append(action)
    return possible

def result(state, action):
    moves = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}
    dx, dy = moves[action]
    return (state[0] + dx, state[1] + dy)

def goal_test(state):
    return state == goal

def reconstruct_path(node):
    path = []
    while node:
        path.append(node.state)
        node = node.parent
    path.reverse()
    return path

def heuristic_dist(state):
    # Manhattan distance: |x1 - x2| + |y1 - y2|
    return abs(state[0] - goal[0]) + abs(state[1] - goal[1])

# -------------------------------
# DFS
# -------------------------------
def dfs_steps():
    frontier = [Node(start, cost=0)]
    explored = set()
    while True:
        if not frontier: return
        node = frontier.pop()
        yield node.state, len(explored), len(frontier), node.cost
        if goal_test(node.state):
            yield reconstruct_path(node), "path"
            return
        explored.add(node.state)
        for action in actions(node.state):
            child_state = result(node.state, action)
            if child_state not in explored and all(child_state != n.state for n in frontier):
                frontier.append(Node(child_state, node, action, node.cost + 1))

# -------------------------------
# BFS
# -------------------------------
def bfs_steps():
    frontier = deque([Node(start, cost=0)])
    explored = set()
    while True:
        if not frontier: return
        node = frontier.popleft()
        yield node.state, len(explored), len(frontier), node.cost
        if goal_test(node.state):
            yield reconstruct_path(node), "path"
            return
        explored.add(node.state)
        for action in actions(node.state):
            child_state = result(node.state, action)
            if child_state not in explored and all(child_state != n.state for n in frontier):
                frontier.append(Node(child_state, node, action, node.cost + 1))

# -------------------------------
# A* Search
# -------------------------------
def astar_steps():
    # Tie-breaker counter
    count = 0
    # Priority Queue stores: (f_score, count, node)
    frontier = [(heuristic_dist(start), count, Node(start, cost=0))]
    explored = set()
    
    while frontier:
        f, _, node = heapq.heappop(frontier)
        if node.state in explored: continue
        
        yield node.state, len(explored), len(frontier), node.cost
        
        if goal_test(node.state):
            yield reconstruct_path(node), "path"
            return
            
        explored.add(node.state)
        for action in actions(node.state):
            child_state = result(node.state, action)
            if child_state not in explored:
                count += 1
                g_cost = node.cost + 1
                f_cost = g_cost + heuristic_dist(child_state)
                heapq.heappush(frontier, (f_cost, count, Node(child_state, node, action, g_cost)))

# -------------------------------
# Greedy Best-First Search
# -------------------------------
def greedy_steps():
    count = 0
    # Priority Queue stores: (h_score, count, node)
    frontier = [(heuristic_dist(start), count, Node(start, cost=0))]
    explored = set()
    
    while frontier:
        h, _, node = heapq.heappop(frontier)
        if node.state in explored: continue
        
        yield node.state, len(explored), len(frontier), node.cost
        
        if goal_test(node.state):
            yield reconstruct_path(node), "path"
            return
            
        explored.add(node.state)
        for action in actions(node.state):
            child_state = result(node.state, action)
            if child_state not in explored:
                count += 1
                heapq.heappush(frontier, (heuristic_dist(child_state), count, Node(child_state, node, action, node.cost + 1)))

# -------------------------------
# Animation
# -------------------------------
def animate_solver(algorithm="BFS"):
    fig, ax = plt.subplots()
    ax.set_facecolor("lightgray")
    maze_img = np.copy(grid)
    img = ax.imshow(maze_img, cmap="gray_r")

    ax.scatter(start[1], start[0], c="green", s=100)
    ax.scatter(goal[1], goal[0], c="red", s=100)

    frontier_text = ax.text(0, -1, "", fontsize=10)
    explored_text = ax.text(10, -1, "", fontsize=10)

    # Select step generator based on algorithm string
    if algorithm == "BFS":
        steps = bfs_steps()
    elif algorithm == "DFS":
        steps = dfs_steps()
    elif algorithm == "ASTAR":
        steps = astar_steps()
    elif algorithm == "GREEDY":
        steps = greedy_steps()
    else:
        steps = bfs_steps()

    def update(frame):
        nonlocal maze_img
        if isinstance(frame[1], str): # Final Path
            for x, y in frame[0]:
                maze_img[x][y] = 0.9
        else: # Exploration step
            state, explored_size, frontier_size, cost = frame
            x, y = state
            maze_img[x][y] = 0.5
            frontier_text.set_text(f"Frontier: {frontier_size}")
            explored_text.set_text(f"Explored: {explored_size}")

        img.set_data(maze_img)
        return [img]

    ani = animation.FuncAnimation(fig, update, frames=steps, interval=200, repeat=False)
    plt.show()

# -------------------------------
# Run
# -------------------------------
if __name__ == "__main__":
    # Options: "DFS", "BFS", "ASTAR", "GREEDY"
    animate_solver("ASTAR")