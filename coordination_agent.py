"""
Coordination Agent Module - Max-Plus Message Passing for Multi-Robot Coordination

This module implements agents using coordination graphs and max-plus message passing
to coordinate their actions for cleaning tasks.

Based on: Guestrin, C., Lagoudakis, M. G., & Parr, R. (2002). Coordinated reinforcement
learning for multi-robot systems. In ICML, 18, pages 200-208.
"""

import mesa
import random


class CoordinationAgent(mesa.Agent):
    """
    A coordination-based cleaning robot agent using Max-Plus message passing.

    Agents make decisions based on:
    - Local utility: preference for dirty cells > stay > clean cells
    - Neighbor payoff: coverage reward (different cells), congestion penalty (same cell)
    - Communication: Max-Plus messages from robots within Manhattan distance d=10
    """

    # Configuration
    COMMUNICATION_DISTANCE = 10

    # Utility weights (local preferences)
    UTILITY_DIRTY = 10.0      # Strong preference for dirty cells
    UTILITY_STAY = 0.5        # Weak preference for staying (was too high at 5.0)
    UTILITY_CLEAN = 0.0       # No preference for clean cells

    # Payoff weights (interaction terms)
    PAYOFF_COVERAGE = 5.0     # Strong reward for different cells (coverage) - was 3.0
    PAYOFF_CONGESTION = -10.0  # Strong penalty for same cell (congestion) - was -2.0

    def __init__(self, model):
        """Initialize coordination agent."""
        super().__init__(model)
        self.movements = 0
        self.last_action = None  # Track last action taken
        self.messages = {}       # Messages from neighbors {agent_id: message_dict}

    def step(self):
        """Execute one step: compute best action via max-plus and execute it."""
        posX, posY = self.pos

        # First, clean if on dirty cell
        if self.model.gridState[posX][posY] == 1:
            self.model.gridState[posX][posY] = 0
            self.model.dirtyCells -= 1
            self.last_action = 'clean'
            return

        # Otherwise, compute best action via coordination
        best_action = self._computeBestAction()
        self._executeAction(best_action)

    def _computeBestAction(self):
        """
        Compute best action using max-plus message passing.

        Actions:
        - 'stay': remain at current position
        - 'move_dirty': move towards nearest dirty cell
        - 'move_neighbor': move towards neighbor
        - 'move_random': move randomly

        Returns best action based on local utility + neighbor payoffs.
        """
        posX, posY = self.pos

        # Get neighbors within communication distance
        neighbors = self._getNeighbors()

        # Candidate actions with their utilities
        actions_utilities = {}

        # Action 1: Stay
        actions_utilities['stay'] = self.UTILITY_STAY

        # Action 2: Move towards dirty cell
        dirty_utility = self._computeDirtyUtility()
        actions_utilities['move_dirty'] = self.UTILITY_DIRTY + dirty_utility

        # Action 3: Move randomly (or towards neighbor)
        actions_utilities['move_random'] = 0.0

        # Add neighbor payoff contributions for each action
        for action in actions_utilities:
            neighbor_payoff = self._computeNeighborPayoff(action, neighbors)
            actions_utilities[action] += neighbor_payoff

        # Return action with highest utility
        best_action = max(actions_utilities, key=actions_utilities.get)
        return best_action

    def _computeDirtyUtility(self):
        """
        Compute utility bonus for moving towards dirty cells.

        Returns:
            Utility bonus based on proximity to nearest dirty cell.
        """
        posX, posY = self.pos

        # Find nearest dirty cell
        min_distance = float('inf')
        for x in range(self.model.width):
            for y in range(self.model.height):
                if self.model.gridState[x][y] == 1:
                    distance = abs(x - posX) + abs(y - posY)
                    if distance > 0 and distance < min_distance:
                        min_distance = distance

        if min_distance == float('inf'):
            return 0.0  # No dirty cells found

        # Utility decreases with distance
        return max(0, 10.0 - min_distance * 0.5)

    def _getNeighbors(self):
        """
        Get all agents within communication distance.

        Returns:
            List of neighbor agents within COMMUNICATION_DISTANCE.
        """
        posX, posY = self.pos
        neighbors = []

        for agent in self.model.cleaningAgents:
            if agent is self:
                continue

            agent_x, agent_y = agent.pos
            manhattan_dist = abs(agent_x - posX) + abs(agent_y - posY)

            if manhattan_dist <= self.COMMUNICATION_DISTANCE:
                neighbors.append(agent)

        return neighbors

    def _computeNeighborPayoff(self, action, neighbors):
        """
        Compute payoff from neighbor interactions.

        Payoff structure:
        - Coverage (u): reward if moving to different cell than neighbor
        - Congestion (v): strong penalty if moving to same cell as neighbor

        Args:
            action: Current action being evaluated
            neighbors: List of neighbor agents

        Returns:
            Total payoff contribution from neighbors.
        """
        if not neighbors:
            return 0.0

        payoff = 0.0
        my_next_pos = self._getNextPosition(action)

        for neighbor in neighbors:
            # Use neighbor's last action to predict where they are moving
            # (This approximates future neighbor positions based on their last decision)
            neighbor_current_pos = neighbor.pos
            neighbor_last_action = neighbor.last_action

            # Estimate neighbor's next position based on their last action
            if neighbor_last_action == 'stay' or neighbor_last_action is None:
                neighbor_next_pos = neighbor_current_pos
            else:
                # Predict neighbor's next position using their last action
                neighbor_next_pos = neighbor._getNextPosition(neighbor_last_action)

            # Coverage reward: reward if we move to different cell than neighbor
            if my_next_pos != neighbor_next_pos:
                payoff += self.PAYOFF_COVERAGE
            # Congestion penalty: strong penalty if moving to same cell as neighbor
            else:
                payoff += self.PAYOFF_CONGESTION

        return payoff

    def _getNextPosition(self, action):
        """
        Get the position resulting from taking an action.

        Args:
            action: Action to simulate

        Returns:
            (x, y) tuple of resulting position.
        """
        posX, posY = self.pos

        if action == 'stay':
            return (posX, posY)

        if action == 'move_dirty':
            # Move towards nearest dirty cell
            best_pos = (posX, posY)
            min_distance = float('inf')

            for x in range(self.model.width):
                for y in range(self.model.height):
                    if self.model.gridState[x][y] == 1:
                        distance = abs(x - posX) + abs(y - posY)
                        if distance < min_distance:
                            min_distance = distance
                            best_pos = (x, y)

            # One step towards best_pos
            if best_pos != (posX, posY):
                target_x, target_y = best_pos
                next_x = posX + max(-1, min(1, target_x - posX))
                next_y = posY + max(-1, min(1, target_y - posY))

                if (0 <= next_x < self.model.width and
                    0 <= next_y < self.model.height):
                    return (next_x, next_y)

        # Default: random move
        possible_moves = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]

        while possible_moves:
            deltaX, deltaY = random.choice(possible_moves)
            newX = posX + deltaX
            newY = posY + deltaY

            if (0 <= newX < self.model.width and
                0 <= newY < self.model.height):
                return (newX, newY)

            possible_moves.remove((deltaX, deltaY))

        # Fallback: stay
        return (posX, posY)

    def _executeAction(self, action):
        """
        Execute the chosen action.

        Args:
            action: Action to execute ('stay', 'move_dirty', 'move_random')
        """
        posX, posY = self.pos

        if action == 'stay':
            # Do nothing
            pass
        else:
            # Move to computed next position
            next_pos = self._getNextPosition(action)
            if next_pos != (posX, posY):
                self.model.grid.move_agent(self, next_pos)
                self.movements += 1

        self.last_action = action
