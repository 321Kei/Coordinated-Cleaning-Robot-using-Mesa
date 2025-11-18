"""
Cleaning Robot Simulation Runner
This script runs experiments with different numbers of agents and analyzes the results.

Author(s): - Ximena Silva Bárcena A01785518
            - Ana Keila Martínez Moreno A01666624
            - Arturo Utrilla Hernández A01174331
            - Rodrigo Martínez Vallejo - A00573055  
Date: 2025-11-11
"""

import random
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
from cleaning_model import CleaningModel


def runSingleSimulation(width, height, numAgents, numDirty, maxTime, agentType='random', seed=None):
    """
    Run a single simulation with the given parameters.

    Args:
        width: Grid width
        height: Grid height
        numAgents: Number of cleaning agents
        numDirty: Number of dirty cells
        maxTime: Maximum simulation time
        agentType: Type of agent ('random' or 'coordination')
        seed: Random seed for reproducibility

    Returns:
        Dictionary with simulation results
    """
    if seed is not None:
        random.seed(seed)

    # Create and run model
    model = CleaningModel(width, height, numAgents, numDirty, maxTime, agentType=agentType)

    # Run until completion or max time
    while model.running:
        model.step()

    # Get results
    results = model.getResults()
    results['numAgents'] = numAgents
    results['agentType'] = agentType

    return results


def runExperiments(width=20, height=30, numDirty=100, maxTime=500,
                   agentCounts=[1, 2, 4, 8, 16], numRepetitions=100, agentTypes=['random', 'coordination']):
    """
    Run multiple experiments with different numbers of agents and strategies.

    Args:
        width: Grid width
        height: Grid height
        numDirty: Number of dirty cells
        maxTime: Maximum simulation time
        agentCounts: List of agent counts to test
        numRepetitions: Number of repetitions per configuration
        agentTypes: List of agent types to test

    Returns:
        DataFrame with all experiment results
    """
    allResults = []

    print(f"Running experiments on {width}x{height} grid with {numDirty} dirty cells")
    print(f"Agent counts to test: {agentCounts}")
    print(f"Agent types to test: {agentTypes}")
    print(f"Repetitions per configuration: {numRepetitions}\n")

    for agentType in agentTypes:
        print(f"\nTesting {agentType.upper()} agents")
        for numAgents in agentCounts:
            print(f"Testing with {numAgents} agent(s)")

            for rep in range(numRepetitions):
                # Run simulation with different seed for each repetition
                results = runSingleSimulation(
                    width, height, numAgents, numDirty, maxTime,
                    agentType=agentType,
                    seed=hash((agentType, numAgents, rep)) % (2**32)
                )
                results['repetition'] = rep
                allResults.append(results)

            print(f"  Completed {numRepetitions} repetitions")

    # Create DataFrame
    df = pd.DataFrame(allResults)
    return df


def analyzeResults(df):
    """
    Analyze and print summary statistics from experiment results.

    Args:
        df: DataFrame with experiment results
    """
    print("EXPERIMENT RESULTS SUMMARY")


    # Analyze by agent type
    for agentType in df['agentType'].unique():
        print(f"\n{agentType.upper()} AGENTS")
        print("-" * 80)

        df_type = df[df['agentType'] == agentType]

        # Group by number of agents
        grouped = df_type.groupby('numAgents')

        # Calculate summary statistics
        summary = grouped.agg({
            'stepsToClean': ['mean', 'std', 'min', 'max'],
            'totalMovements': ['mean', 'std', 'min', 'max'],
            'cleanPercentage': ['mean', 'std']
        }).round(2)

        print("AVERAGE RESULTS BY NUMBER OF AGENTS")
        print(summary)

        # Check which configurations achieved 100% cleaning within 500 steps
        success_rates = grouped.apply(
            lambda x: ((x['cleanPercentage'] == 100) & (x['stepsToClean'] <= 500)).sum() / len(x) * 100,
            include_groups=False
        )

        print("\nSUCCESS RATE (100% CLEAN WITHIN 500 STEPS):")
        for agents, pct in success_rates.items():
            print(f"  {agents} agent(s): {pct:.1f}%")

        print("\n")


def createVisualizations(df, outputPrefix='cleaning_simulation'):
    """
    Create visualization plots from experiment results.

    Args:
        df: DataFrame with experiment results
        outputPrefix: Prefix for output file names
    """
    # Set style and color palette
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (18, 12)

    # Define consistent colors: blue for random, orange for coordination
    COLOR_RANDOM = '#1f77b4'  # Blue
    COLOR_COORDINATION = '#ff7f0e'  # Orange
    color_palette = {'random': COLOR_RANDOM, 'coordination': COLOR_COORDINATION}

    # Create figure with subplots
    fig, axes = plt.subplots(3, 2, figsize=(18, 15))

    # Calculate success rates (100% clean within 500 steps)
    df['success'] = (df['cleanPercentage'] == 100) & (df['stepsToClean'] <= 500)

    # 1. Steps to clean vs Number of agents (by agent type)
    ax1 = axes[0, 0]
    dfClean = df[df['stepsToClean'].notna()]
    sns.boxplot(data=dfClean, x='numAgents', y='stepsToClean', hue='agentType', ax=ax1,
                palette=color_palette)
    ax1.set_xlabel('Number of Agents', fontsize=12)
    ax1.set_ylabel('Steps to Clean All Cells', fontsize=12)
    ax1.set_title('Time Required to Clean All Cells', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # 2. Total movements vs Number of agents (by agent type)
    ax2 = axes[0, 1]
    sns.boxplot(data=df, x='numAgents', y='totalMovements', hue='agentType', ax=ax2,
                palette=color_palette)
    ax2.set_xlabel('Number of Agents', fontsize=12)
    ax2.set_ylabel('Total Movements', fontsize=12)
    ax2.set_title('Total Movements by All Agents', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # 3. Average steps to clean (line plot by agent type)
    ax3 = axes[1, 0]
    for agentType in sorted(df['agentType'].unique()):
        df_type = dfClean[dfClean['agentType'] == agentType]
        avgSteps = df_type.groupby('numAgents')['stepsToClean'].mean()
        stdSteps = df_type.groupby('numAgents')['stepsToClean'].std()
        ax3.plot(avgSteps.index, avgSteps.values, marker='o', linewidth=2.5,
                markersize=8, label=agentType, color=color_palette[agentType])
        ax3.fill_between(avgSteps.index,
                        avgSteps.values - stdSteps.values,
                        avgSteps.values + stdSteps.values,
                        alpha=0.2, color=color_palette[agentType])
    ax3.set_xlabel('Number of Agents', fontsize=12)
    ax3.set_ylabel('Average Steps to Clean', fontsize=12)
    ax3.set_title('Average Time to Complete Cleaning', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Average movements per agent (by agent type)
    ax4 = axes[1, 1]
    df['movementsPerAgent'] = df['totalMovements'] / df['numAgents']
    for agentType in sorted(df['agentType'].unique()):
        df_type = df[df['agentType'] == agentType]
        avgMovementsPerAgent = df_type.groupby('numAgents')['movementsPerAgent'].mean()
        ax4.plot(avgMovementsPerAgent.index, avgMovementsPerAgent.values,
                marker='o', linewidth=2.5, markersize=8, label=agentType,
                color=color_palette[agentType])
    ax4.set_xlabel('Number of Agents', fontsize=12)
    ax4.set_ylabel('Average Movements per Agent', fontsize=12)
    ax4.set_title('Efficiency: Movements per Agent', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. SUCCESS RATE HISTOGRAM (100% clean within 500 steps)
    ax5 = axes[2, 0]
    success_data = []
    agent_labels = []
    colors = []

    for agentType in sorted(df['agentType'].unique()):
        for numAgents in sorted(df['numAgents'].unique()):
            df_subset = df[(df['agentType'] == agentType) & (df['numAgents'] == numAgents)]
            success_rate = df_subset['success'].sum() / len(df_subset) * 100
            success_data.append(success_rate)
            agent_labels.append(f"{agentType[:3].upper()}\n{numAgents}")
            colors.append(color_palette[agentType])

    ax5.bar(range(len(success_data)), success_data, color=colors)
    ax5.set_xticks(range(len(success_data)))
    ax5.set_xticklabels(agent_labels, fontsize=10)
    ax5.set_ylabel('Success Probability (%)', fontsize=12)
    ax5.set_title('Success Rate: 100% Clean within 500 Steps', fontsize=14, fontweight='bold')
    ax5.set_ylim([0, 105])
    ax5.grid(True, alpha=0.3, axis='y')

    # Add legend for colors
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=COLOR_RANDOM, label='Random Walk'),
                      Patch(facecolor=COLOR_COORDINATION, label='Coordination')]
    ax5.legend(handles=legend_elements, loc='upper left')

    # 6. Success rate line plot by agent type
    ax6 = axes[2, 1]
    for agentType in sorted(df['agentType'].unique()):
        success_rates = []
        agent_counts = sorted(df['numAgents'].unique())
        for numAgents in agent_counts:
            df_subset = df[(df['agentType'] == agentType) & (df['numAgents'] == numAgents)]
            success_rate = df_subset['success'].sum() / len(df_subset) * 100
            success_rates.append(success_rate)
        ax6.plot(agent_counts, success_rates, marker='o', linewidth=2.5,
                markersize=8, label=agentType, color=color_palette[agentType])

    ax6.set_xlabel('Number of Agents', fontsize=12)
    ax6.set_ylabel('Success Probability (%)', fontsize=12)
    ax6.set_title('Success Rate vs Number of Agents', fontsize=14, fontweight='bold')
    ax6.set_ylim([0, 105])
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    outputFile = f'{outputPrefix}_analysis.png'
    plt.savefig(outputFile, dpi=300, bbox_inches='tight')
    

    return fig


def saveResultsToCsv(df, filename='cleaning_simulation_results.csv'):
    """
    Save experiment results to CSV file.

    Args:
        df: DataFrame with experiment results
        filename: Output CSV filename
    """
    df.to_csv(filename, index=False)
    print(f"Results saved to: {filename}")


def main():
    """
    Main function to run experiments and generate analysis.
    """

    # Experiment parameters (as specified: 20x30 grid, 100 dirty cells, 500 max steps)
    GRID_WIDTH = 20
    GRID_HEIGHT = 30
    NUM_DIRTY = 100
    MAX_TIME = 500
    AGENT_COUNTS = [1, 2, 4, 8, 16]
    NUM_REPETITIONS = 100
    AGENT_TYPES = ['random', 'coordination']

    # Run experiments
    resultsDf = runExperiments(
        width=GRID_WIDTH,
        height=GRID_HEIGHT,
        numDirty=NUM_DIRTY,
        maxTime=MAX_TIME,
        agentCounts=AGENT_COUNTS,
        numRepetitions=NUM_REPETITIONS,
        agentTypes=AGENT_TYPES
    )

    # Analyze results
    analyzeResults(resultsDf)

    # Create visualizations
    createVisualizations(resultsDf)

    # Save results
    saveResultsToCsv(resultsDf)


if __name__ == "__main__":
    main()