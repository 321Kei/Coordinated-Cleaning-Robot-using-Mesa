"""
Cleaning Robot Agent Module
"""

import mesa
import random


class CleaningAgent(mesa.Agent):
    """A reactive cleaning robot agent"""
    
    def __init__(self, model):
        super().__init__(model)
        self.movements = 0
    
    def step(self):
        """Execute one step: clean if dirty, otherwise move randomly"""
        posX, posY = self.pos

        if self.model.gridState[posX][posY] == 1:
            # Clean the cell
            self.model.gridState[posX][posY] = 0
            self.model.dirtyCells -= 1
        else:
            # Move randomly
            self._moveRandomly()
    
    def _moveRandomly(self):
        """Move to a random neighboring cell (8-neighborhood)"""
        possibleMoves = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]

        deltaX, deltaY = random.choice(possibleMoves)
        currentX, currentY = self.pos
        newX = currentX + deltaX
        newY = currentY + deltaY

        if (0 <= newX < self.model.width and
            0 <= newY < self.model.height):
            self.model.grid.move_agent(self, (newX, newY))
            self.movements += 1
