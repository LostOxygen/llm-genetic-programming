"""library for genetic algorithm operations"""

import torch
import random
from ga.chromosome import Chromosome
from ga.population import Population

def traversal(start_pos: int, chromosome: Chromosome) -> int:
    """
    Function to traverse the tree from the given position

    Parameters:
        start_pos: int - the start position
        chromosome: Chromosome - the chromosome to be traversed

    Returns:
        int - the new position
    """
    if chromosome.gen[start_pos] in chromosome.terminal_set:
        return start_pos + 1
    elif chromosome.gen[start_pos] in chromosome.func_set[1]:
        return traversal(start_pos + 1, chromosome)
    else:
        new_pos = traversal(start_pos + 1, chromosome)
        return traversal(new_pos, chromosome)


def mutate(chromosome: Chromosome) -> Chromosome:
    """
    Function to mutate a chromosome

    Parameters:
        chromosome: Chromosome - the chromosome to be mutated

    Returns:
        Chromosome - the mutated chromosome
    """
    poz = torch.random.randint(len(chromosome.gen))
    if chromosome.gen[poz] in chromosome.func_set[1] + chromosome.func_set[2]:
        if chromosome.gen[poz] in chromosome.func_set[1]:
            chromosome.gen[poz] = random.choice(chromosome.func_set[1])
        else:
            chromosome.gen[poz] = random.choice(chromosome.func_set[2])
    else:
        chromosome.gen[poz] = random.choice(chromosome.terminal_set)
    return chromosome


def selection(population: Population, num_sel: int) -> Chromosome:
    """
    Function to select a member of the population for crossing over

    Parameters:
        population: Population - population of chromosomes
        num_sel: int - number of chromosome selected from the population

    Returns:
        Chromosome - the selected chromosome
    """
    sample = random.sample(population.list, num_sel)
    best = sample[0]
    for i in range(1, len(sample)):
        if population.list[i].fitness < best.fitness:
            best = population.list[i]

    return best


def cross_over(mommy: Chromosome, daddy: Chromosome, max_depth: int) -> Chromosome:
    """
    Function to cross over two chromosomes in order to obtain a child

    Parameters:
        mommy: Chromosome - mommy chromosome to be crossed over
        daddy: Chromosome - daddy chromosome to be crossed over
        max_depth: int - maximum depth of a tree

    Returns:
        Chromosome - the new child chromosome
    """
    child = Chromosome(mommy.terminal_set, mommy.func_set, mommy.depth, None)
    start_m = torch.random.randint(len(mommy.gen))
    start_f = torch.random.randint(len(daddy.gen))
    end_m = traversal(start_m, mommy)
    end_f = traversal(start_f, daddy)
    child.gen = mommy.gen[:start_m] + daddy.gen[start_f : end_f] + mommy.gen[end_m :]
    if child.get_depth() > max_depth and random.random() > 0.2:
        child = Chromosome(mommy.terminal_set, mommy.func_set, mommy.depth)
    return child


def get_best(population: Population) -> Chromosome:
    """
    Function to get the best chromosome from the population

    Parameters:
        population: Population - population to get the best chromosome from

    Returns:
        Chromosome - best chromosome from population
    """
    best = population.list[0]
    for i in range(1, len(population.list)):
        if population.list[i].fitness < best.fitness:
            best = population.list[i]

    return best


def get_worst(population: Population) -> Chromosome:
    """
    Function to get the worst chromosome of the population

    Parameters:
        population: Population - the population to get the worst chromosome from

    Returns:
        Chromosome - worst chromosome from the population
    """
    worst = population.list[0]
    for i in range(1, len(population.list)):
        if population.list[i].fitness > worst.fitness:
            worst = population.list[i]

    return worst


def replace_worst(population: Population, chromosome: Chromosome) -> Population:
    """
    Function to replace the worst chromosome of the population with a new one

    Parameters:
        population: Population - population
        chromosome: Chromosome - chromosome to be added

    Returns:
        Population - the updated population
    """
    worst = get_worst(population)
    if chromosome.fitness < worst.fitness:
        for i in range(len(population.list)):
            if population.list[i].fitness == worst.fitness:
                population.list[i] = chromosome
                break

    return population
