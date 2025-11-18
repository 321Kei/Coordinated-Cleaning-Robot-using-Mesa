"""
Cleaning Robot Model Module
This module defines the CleaningModel class for multi-agent cleaning simulation.
"""

import mesa
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
import random
from cleaning_agent import CleaningAgent


class CleaningModel(mesa.Model):
    """
    A model representing a room with cleaning robots.

    The model simulates multiple cleaning agents in an MxN grid where some cells
    are initially dirty. All agents start at position [1,1]. Agents clean dirty
    cells and move randomly when on clean cells.

    Attributes:
        width: Width of the grid (M)
        height: Height of the grid (N)
        numAgents: Number of cleaning agents
        dirtyPercentage: Initial percentage of dirty cells (0-100)
        maxTime: Maximum number of steps to run the simulation
        grid: Mesa MultiGrid for agent positioning
        gridState: 2D list representing clean (0) or dirty (1) cells
        dirtyCells: Counter for remaining dirty cells
        stepsToClean: Number of steps taken until all cells were clean
        running: Boolean indicating if the model should continue running
    """
    
    def __init__(self, width, height, numAgents, numDirty, maxTime, agentType='random'):
        """
        Initialize the cleaning model.

        Args:
            width: Width of the grid (M)
            height: Height of the grid (N)
            numAgents: Number of cleaning agents
            numDirty: Number of dirty cells
            maxTime: Maximum number of steps to run
            agentType: Type of agent ('random' or 'coordination')
        """
        super().__init__()
        self.width = width
        self.height = height
        self.numAgents = numAgents
        self.numDirty = numDirty
        self.maxTime = maxTime
        self.agentType = agentType
        self.currentStep = 0
        self.stepsToClean = None

        # Create grid (allows multiple agents per cell)
        self.grid = MultiGrid(width, height, torus=False)

        # Initialize grid state
        self.gridState = [[0 for _ in range(height)] for _ in range(width)]

        self.dirtyCells = numDirty
        self.initialDirtyCells = numDirty

        # Randomly select cells to be dirty
        allPositions = [(x, y) for x in range(width) for y in range(height)]
        dirtyPositions = random.sample(allPositions, numDirty)

        for x, y in dirtyPositions:
            self.gridState[x][y] = 1

        # Create agents and place them at random clean cells
        self.cleaningAgents = []
        cleanPositions = [p for p in allPositions if p not in dirtyPositions]

        # Ensure there are clean positions for agents
        if not cleanPositions:
            raise ValueError(f"No clean positions available! numDirty ({numDirty}) >= total cells ({width * height})")

        for i in range(numAgents):
            if agentType == 'coordination':
                from coordination_agent import CoordinationAgent
                agent = CoordinationAgent(self)
            else:
                agent = CleaningAgent(self)

            self.cleaningAgents.append(agent)
            # Place at random clean cell
            start_pos = random.choice(cleanPositions)
            self.grid.place_agent(agent, start_pos)

        self.running = True

        # Data collector for statistics
        self.dataCollector = DataCollector(
            model_reporters={
                "DirtyCells": lambda m: m.dirtyCells,
                "CleanPercentage": lambda m: ((m.width * m.height - m.dirtyCells) /
                                              (m.width * m.height)) * 100,
                "TotalMovements": lambda m: sum(a.movements for a in m.cleaningAgents)
            }
        )
    
    def step(self):
        """
        Execute one step of the model:
        - Activate all agents
        - Collect data
        - Check termination conditions
        """
        self.currentStep += 1

        # Activate all agents in random order
        agentsShuffled = self.cleaningAgents.copy()
        random.shuffle(agentsShuffled)
        for agent in agentsShuffled:
            agent.step()
        
        # Collect data
        self.dataCollector.collect(self)
        
        # Check if all cells are clean
        if self.dirtyCells == 0 and self.stepsToClean is None:
            self.stepsToClean = self.currentStep
        
        # Check if max time reached
        if self.currentStep >= self.maxTime:
            self.running = False
    
    def getTotalMovements(self):
        """
        Calculate total movements made by all agents.
        
        Returns:
            Total number of movements across all agents
        """
        return sum(agent.movements for agent in self.cleaningAgents)
    
    def getCleanPercentage(self):
        """
        Calculate the percentage of clean cells.

        Returns:
            Percentage of clean cells (0-100)
        """
        totalCells = self.width * self.height
        cleanCells = totalCells - self.dirtyCells
        return (cleanCells / totalCells) * 100

    def getResults(self):
        """
        Get simulation results.

        Returns:
            Dictionary containing:
                - stepsToClean: Steps until all cells clean (or None)
                - cleanPercentage: Final percentage of clean cells
                - totalMovements: Total movements by all agents
                - maxTimeReached: Whether max time was reached
        """
        return {
            'stepsToClean': self.stepsToClean,
            'cleanPercentage': self.getCleanPercentage(),
            'totalMovements': self.getTotalMovements(),
            'maxTimeReached': self.currentStep >= self.maxTime,
            'totalSteps': self.currentStep
        }