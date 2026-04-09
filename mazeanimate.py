import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import heapq


# Simple maze: 1 = free, 0 = wall
MAZE = np.array([
    [1, 0, 1, 1, 1, 1, 1, 1],
    [1, 0, 1, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 1, 0, 1],
    [1, 1, 1, 1, 0, 1, 1, 1],
    [0, 0, 0, 1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
], dtype=int)

ROWS, COLS = MAZE.shape

# Start and goal coordinates (row, col)
start = (0, 0)
goal = (6, 7)

# Image grid: use floats 0.0..1.0 where 0=wall (black), 1=free (white)
grid = MAZE.astype(float)


class Node:
    def __init__(self, state, parent=None, action=None, cost=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.cost = cost


class PriorityNode(Node):
    def __init__(self, state, parent=None, action=None, cost=0, priority=0):
        super().__init__(state, parent, action, cost)
        self.priority = priority

    def __lt__(self, other):
        return self.priority < other.priority


def goal_test(state):
    return state == goal


def actions(state):
    r, c = state
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < ROWS and 0 <= nc < COLS and grid[nr, nc] == 1.0:
            yield (dr, dc)


def result(state, action):
    r, c = state
    dr, dc = action
    return (r + dr, c + dc)


def reconstruct_path(node):
    path = []
    cur = node
    while cur is not None:
        path.append(cur.state)
        cur = cur.parent
    path.reverse()
    return path


def heuristic_func(state):
    return abs(state[0] - goal[0]) + abs(state[1] - goal[1])


def greedy_bfs_steps():
    """Greedy Best-First Search: priority = heuristic(state)"""
    start_node = PriorityNode(start, cost=0, priority=heuristic_func(start))
    frontier = []
    heapq.heappush(frontier, start_node)
    explored = set()

    while frontier:
        node = heapq.heappop(frontier)

        yield node.state, len(explored), len(frontier), node.cost

        if goal_test(node.state):
            yield reconstruct_path(node), "path"
            return

        if node.state in explored:
            continue

        explored.add(node.state)

        for act in actions(node.state):
            child_state = result(node.state, act)
            if child_state in explored:
                continue
            # avoid duplicates in frontier
            if any(child_state == n.state for n in frontier):
                continue
            h = heuristic_func(child_state)
            child = PriorityNode(child_state, node, act, node.cost + 1, priority=h)
            heapq.heappush(frontier, child)


def a_star_steps():
    """A* search: priority = g + h"""
    start_node = PriorityNode(start, cost=0, priority=heuristic_func(start))
    frontier = []
    heapq.heappush(frontier, start_node)
    explored = {}  # state -> best g

    while frontier:
        node = heapq.heappop(frontier)

        yield node.state, len(explored), len(frontier), node.cost

        if goal_test(node.state):
            yield reconstruct_path(node), "path"
            return

        # skip if we've already found a better path
        if node.state in explored and node.cost > explored[node.state]:
            continue

        explored[node.state] = node.cost

        for act in actions(node.state):
            child_state = result(node.state, act)
            new_g = node.cost + 1
            # if we've seen a better g for child, skip
            if child_state in explored and new_g >= explored[child_state]:
                continue
            new_f = new_g + heuristic_func(child_state)
            child = PriorityNode(child_state, node, act, new_g, priority=new_f)
            heapq.heappush(frontier, child)


def animate_solver(algorithm="GREEDY"):
    fig, ax = plt.subplots()
    ax.set_title(f"AI {algorithm} Maze Solver")
    ax.set_facecolor("lightgray")

    maze_img = np.copy(grid)
    img = ax.imshow(maze_img, cmap="gray_r", vmin=0, vmax=1)

    ax.scatter(start[1], start[0], c="green", s=100, label="Start")
    ax.scatter(goal[1], goal[0], c="red", s=100, label="Goal")

    frontier_text = ax.text(0.01, 0.98, "", transform=ax.transAxes, fontsize=10, va='top')
    explored_text = ax.text(0.5, 0.98, "", transform=ax.transAxes, fontsize=10, va='top')

    algos = {
        "GREEDY": greedy_bfs_steps,
        "ASTAR": a_star_steps,
    }

    steps = algos.get(algorithm.upper())
    if steps is None:
        raise ValueError("Algorithm must be 'GREEDY' or 'ASTAR'")

    steps_gen = steps()

    def update(frame):
        nonlocal maze_img
        if isinstance(frame[1], str):
            path = frame[0]
            for x, y in path:
                maze_img[x][y] = 0.9
        else:
            state, explored_size, frontier_size, cost = frame
            x, y = state
            # don't overwrite start/goal markers; set explored shading
            if (x, y) != start and (x, y) != goal:
                maze_img[x][y] = 0.5
            frontier_text.set_text(f"Frontier: {frontier_size}")
            explored_text.set_text(f"Explored: {explored_size}")

        img.set_data(maze_img)
        return [img]

    ani = animation.FuncAnimation(
        fig, update, frames=steps_gen, interval=80, repeat=False, cache_frame_data=False
    )
    plt.show()


if __name__ == "__main__":
    import sys
    alg = sys.argv[1] if len(sys.argv) > 1 else "ASTAR"
    animate_solver(alg)