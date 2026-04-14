import random
from matplotlib import pyplot as plt

#Noise settings
hasNoise = True
strategyNoise = 0.1  # Chance of making a mistake
payoffNoise = 0.2    # Variation in scoring
randomMutation = 0.02  # Chance of full random mutation


def main():
    # Algorithm parameters
    popSize = 100
    generations = 50
    strategyLength = 3
    crossoverRate = 0.8
    mutationRate = 0.1
    elitism = 2

    #proportions = [0.2, 0.2, 0.2, 0.2, 0.2]  # Equal proportions
    #proportions = [1, 0, 0, 0, 0]  # All Cooperators
    #proportions = [0, 1, 0, 0, 0]  # All Defectors
    #proportions = [0, 0, 1, 0, 0]  # All Tit-for-Tat
    #proportions = [0, 0, 0, 1, 0]  # All Suspicious Tit-for-Tat
    proportions = [0, 0, 0, 0, 1]  # All Random
    print(f"Proportions: {proportions}")

    bestGenome, fitnessHistory = evolveIPD(
        popSize, generations, strategyLength, crossoverRate, mutationRate, elitism, proportions
    )

    print(f"Best strategy genome: {bestGenome}")

    filename = f"fitness_{'with_noise' if hasNoise else 'without_noise'}.png"
    visualiseFitness(fitnessHistory, filename)

    fixedStrategies = {
        "Always Cooperate": alwaysCooperate,
        "Always Defect": alwaysDefect,
        "Tit-for-Tat": titForTat,
        "Suspicious Tit-for-Tat": suspiciousTitForTat,
        "Random": randomStrategy
    }

    bestStrategy = createMemoryStrategy(bestGenome)

    results = {}
    for name, strategy in fixedStrategies.items():
        score1, score2 = playIPD(bestStrategy, strategy)
        results[name] = (score1, score2)

    print(f"\nPerformance against fixed strategies ({'with' if hasNoise else 'without'} noise):")
    for name, (score1, score2) in results.items():
        print(f"{name}: Evolved Score = {score1:.2f}, Fixed Score = {score2:.2f}")


def playIPD(p1Strategy, p2Strategy, rounds=200):
    """
    Play IPD between two strategies.
    """
    base_payoff = {
        ('C', 'C'): (3, 3),
        ('C', 'D'): (0, 5),
        ('D', 'C'): (5, 0),
        ('D', 'D'): (1, 1)
    }

    p1Score, p2Score = 0, 0
    p1History, p2History = [], []

    for _ in range(rounds):
        p1Move = p1Strategy(p2History)
        p2Move = p2Strategy(p1History)

        p1RoundScore, p2RoundScore = base_payoff[(p1Move, p2Move)]

        # Introduce random noise into the payoff (if enabled)
        if hasNoise:
            p1RoundScore += random.uniform(-payoffNoise, payoffNoise)
            p2RoundScore += random.uniform(-payoffNoise, payoffNoise)

            p1RoundScore = max(0, p1RoundScore)
            p2RoundScore = max(0, p2RoundScore)

        p1History.append(p1Move)
        p2History.append(p2Move)

        p1Score += p1RoundScore
        p2Score += p2RoundScore

    return p1Score, p2Score


def alwaysCooperate(opponentHistory):
    return 'C'


def alwaysDefect(opponentHistory):
    return 'D'


def titForTat(opponentHistory):
    if not opponentHistory:
        return 'C'  # Cooperate on first move
    return opponentHistory[-1]  # Copy opponent's last move


def suspiciousTitForTat(opponentHistory):
    if not opponentHistory:
        return 'D'
    return opponentHistory[-1]


def randomStrategy(opponentHistory):
    return random.choice(['C', 'D'])


def createMemoryStrategy(genome):
    """
    Creates a strategy function from a genome that can remember previous moves.
    """
    def strategy(opponentHistory):
        if not opponentHistory:
            move = genome[0]
        else:
            lastMove = opponentHistory[-1]
            move = genome[1] if lastMove == 'C' else genome[2]

        #Add Noise if enabled
        if hasNoise and random.random() < strategyNoise:
            move = 'D' if move == 'C' else 'C'

        return move

    return strategy


def evolveIPD(popSize, generations, strategyLength, crossoverRate, mutationRate, elitism, proportions):
    """
    Evolve strategies using weighted proportions
    """
    fixedStrategies = [
        alwaysCooperate,
        alwaysDefect,
        titForTat,
        suspiciousTitForTat,
        randomStrategy
    ]

    population = initializePopulation(popSize, strategyLength)
    bestFitnessHistory = []

    for gen in range(generations):

        fitnessScores = [
            evaluateFitnessWithProportions(genome, fixedStrategies, proportions)
            for genome in population
        ]

        sortedPopulation = [x for _, x in sorted(zip(fitnessScores, population), key=lambda pair: pair[0], reverse=True)]
        sortedFitness = sorted(fitnessScores, reverse=True)

        bestFitnessHistory.append(sortedFitness[0])

        print(f"Generation {gen + 1}: Best fitness = {sortedFitness[0]:.2f}")

        newPopulation = sortedPopulation[:elitism]

        while len(newPopulation) < popSize:

            parent1 = tournamentSelection(population, fitnessScores)
            parent2 = tournamentSelection(population, fitnessScores)

            if random.random() < crossoverRate:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = parent1[:], parent2[:]

            child1 = mutation(child1, mutationRate)
            child2 = mutation(child2, mutationRate)
            newPopulation.append(child1)
            newPopulation.append(child2)

        population = newPopulation[:popSize]

    bestGenome = sortedPopulation[0]
    return bestGenome, bestFitnessHistory


def initializePopulation(popSize, strategyLength):
    """
    Generate initial population
    """
    population = []
    for _ in range(popSize):
        genome = [random.choice(['C', 'D']) for _ in range(strategyLength)]
        population.append(genome)
    return population


def evaluateFitnessWithProportions(genome, fixedStrategies, proportions, rounds=200):
    """
    Evaluate a genome's fitness against fixed strategies with different proportions
    """
    strategy = createMemoryStrategy(genome)
    totalScore = 0

    for fixedStrategy, proportion in zip(fixedStrategies, proportions):
        score, _ = playIPD(strategy, fixedStrategy, rounds)
        totalScore += score * proportion * 5

    return totalScore


def tournamentSelection(population, fitnessScores, tournamentSize=3):
    """
    Select a parent using tournament selection
    """
    tournamentIndices = random.sample(range(len(population)), tournamentSize)
    tournamentFitness = [fitnessScores[i] for i in tournamentIndices]
    winnerIndex = tournamentIndices[tournamentFitness.index(max(tournamentFitness))]
    return population[winnerIndex]


def crossover(parent1, parent2):
    """
    One-point crossover for strategy genomes
    """
    if len(parent1) <= 1:
        return parent1[:], parent2[:]

    point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]

    return child1, child2


def mutation(genome, mutationRate):
    """
    Apply mutation by flipping bits with mutationRate.
    """
    mutated = genome[:]

    for i in range(len(mutated)):
        if random.random() < mutationRate:
            mutated[i] = 'D' if mutated[i] == 'C' else 'C'

    #Add random mutation
    if hasNoise and random.random() < randomMutation:
        mutated = [random.choice(['C', 'D']) for _ in range(len(genome))]

    return mutated


def analyzeBestStrategy(bestGenome, fixedStrategies):
    """
    Analyse best strategy against each fixed strategy
    """
    bestStrategy = createMemoryStrategy(bestGenome)
    results = {}

    for name, strategy in fixedStrategies.items():
        score1, score2 = playIPD(bestStrategy, strategy)
        results[name] = (score1, score2)

    return results


def visualiseFitness(fitnessHistory, filename="fitness.png"):
    """
    Visualise fitness progression over generations
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(fitnessHistory) + 1), fitnessHistory, marker='o')
    plt.title('Best Fitness Progression')
    plt.xlabel('Generation')
    plt.ylabel('Fitness Score')
    plt.grid(True)
    plt.savefig(filename)
    plt.show()


if __name__ == "__main__":
    main()
