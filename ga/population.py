"""implementation of a population class for the genetic algorithm"""

import random
from ga.chromosome import Chromosome

class Population:
    """
    Class for representing a population of chromosomes
    """

    def __init__(self,
            pop_size: int,
            num_selected: int,
            func_set: dict,
            terminal_set: list,
            depth: int,
            max_depth: int,
        ) -> None:
        """
        Constructor for population class

        Parameters:
            pop_size: int - number of members in the population
            func_set: dict - set of functions for the population
            terminal_set: list - set of terminals for the population
            num_selected: int - number of chromosomes selected from the population
            depth: int - initial depth of a tree
            max_depth: int - maximum depth of a tree

        Returns:
            None
        """
        self.pop_size = pop_size
        self.num_selected = num_selected
        self.list = self.create_population(self.pop_size, func_set, terminal_set, depth)
        self.max_depth = max_depth


    def create_population(self,
            pop_size: int,
            func_set: dict,
            terminal_set: list,
            depth: int
        ) -> list:
        """
        Function to create a population

        Parameters:
            pop_size: int - number of members in the population
            func_set: dict - set of functions for the population
            terminal_set: list - set of terminals for the population
            depth: int - initial depth of a tree
        
        Returns:
            list - list of chromosomes (the population)
        """
        pop_list = []
        for _ in range(pop_size):
            if random.random() > 0.5:
                pop_list.append(Chromosome(terminal_set, func_set, depth, 'grow'))
            else:
                pop_list.append(Chromosome(terminal_set, func_set, depth, 'full'))
        return pop_list
