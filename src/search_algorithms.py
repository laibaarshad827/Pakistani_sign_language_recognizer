
"""BFS, DFS and A* for PSL curriculum navigation."""

import heapq
from collections import deque


def bfs_path(graph, start, goal):
    """Find a path with the fewest edges using Breadth-First Search."""

    if start not in graph or goal not in graph:
        return None

    queue = deque([[start]])
    visited = {start}

    while queue:
        path = queue.popleft()
        current = path[-1]

        if current == goal:
            return path

        for neighbor in graph.get(current, {}):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(path + [neighbor])

    return None


def dfs_path(graph, start, goal):
    """Find a path using Depth-First Search."""

    if start not in graph or goal not in graph:
        return None

    stack = [[start]]
    visited = set()

    while stack:
        path = stack.pop()
        current = path[-1]

        if current == goal:
            return path

        if current in visited:
            continue

        visited.add(current)

        for neighbor in reversed(list(graph.get(current, {}))):
            if neighbor not in visited:
                stack.append(path + [neighbor])

    return None


def a_star_path(graph, start, goal, heuristic=None):
    """Find a minimum-cost route using A* search."""

    if start not in graph or goal not in graph:
        return None

    if heuristic is None:
        heuristic = lambda node: 0

    frontier = []
    heapq.heappush(frontier, (heuristic(start), 0, start))

    came_from = {start: None}
    cost_so_far = {start: 0}

    while frontier:
        _, current_cost, current = heapq.heappop(frontier)

        if current_cost != cost_so_far[current]:
            continue

        if current == goal:
            path = []
            node = goal

            while node is not None:
                path.append(node)
                node = came_from[node]

            return list(reversed(path))

        for neighbor, edge_cost in graph.get(current, {}).items():
            if edge_cost < 0:
                raise ValueError("A* requires non-negative edge costs.")

            new_cost = current_cost + edge_cost

            if (
                neighbor not in cost_so_far
                or new_cost < cost_so_far[neighbor]
            ):
                cost_so_far[neighbor] = new_cost
                came_from[neighbor] = current

                priority = new_cost + heuristic(neighbor)

                heapq.heappush(
                    frontier,
                    (priority, new_cost, neighbor)
                )

    return None


class LearningPathPlanner:
    """Builds search routes from a learner's unlocked curriculum."""

    def __init__(self, agent):
        self.agent = agent
        self.graph = {}
        self.positions = {}
        self.build_graph()

    def build_graph(self):
        """Connect adjacent signs in the currently unlocked curriculum."""

        level_progress = self.agent.get_level_progress()

        unlocked_signs = []

        for level_info in level_progress:
            if level_info["unlocked"]:
                unlocked_signs.extend(
                    self.agent.hierarchy.get_signs(level_info["level"])
                )

        self.graph = {sign: {} for sign in unlocked_signs}
        self.positions = {
            sign: index
            for index, sign in enumerate(unlocked_signs)
        }

        for index in range(len(unlocked_signs) - 1):
            first = unlocked_signs[index]
            second = unlocked_signs[index + 1]

            first_mastery = self.agent.learner.mastery(first)
            second_mastery = self.agent.learner.mastery(second)

            # Higher-mastery signs have a higher navigation cost.
            forward_cost = 1.0 + (4.0 * second_mastery)
            backward_cost = 1.0 + (4.0 * first_mastery)

            self.graph[first][second] = forward_cost
            self.graph[second][first] = backward_cost

    def _validate_signs(self, start, goal):
        if start not in self.graph:
            raise ValueError(
                f"Starting sign '{start}' is not currently unlocked."
            )

        if goal not in self.graph:
            raise ValueError(
                f"Target sign '{goal}' is not currently unlocked."
            )

    def find_bfs_route(self, start, goal):
        self._validate_signs(start, goal)
        return bfs_path(self.graph, start, goal)

    def find_dfs_route(self, start, goal):
        self._validate_signs(start, goal)
        return dfs_path(self.graph, start, goal)

    def find_astar_route(self, start, goal):
        self._validate_signs(start, goal)

        goal_position = self.positions[goal]

        def heuristic(sign):
            # Each edge costs at least 1.
            return abs(self.positions[sign] - goal_position)

        return a_star_path(
            self.graph,
            start,
            goal,
            heuristic
        )
