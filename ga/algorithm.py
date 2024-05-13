"""helper class implementing the genetic algorithm"""

from ga.chromosome import Chromosome
from ga.operations import selection, mutate, cross_over, get_best, replace_worst

class GeneticAlgorithm:
    """
    Genetic Algorithm class
    """
    def __init__(self,
        population: int,
        iterations: int,
        inputs: float,
        labels: float,
        ) -> None:
        """
        Constructor for the GeneticAlgorithm class

        Parameters:
            population: int - the size of the population
            iterations: int - the number of iterations
            inputs: float - the input data (X values)
            labels: float - the labels (y values)

        Returns:
            None
        """
        self.population = population
        self.iterations = iterations
        self.inputs = inputs
        self.outputs = labels

    def __step(self) -> None:
        """
        Private function to do one step of the algorithm
        """

        mother = selection(self.population, self.population.num_selected)
        father = selection(self.population, self.population.num_selected)

        child = cross_over(mother, father, self.population.max_depth)
        child = mutate(child)
        child.calculate_fitness(self.inputs, self.outputs)
        self.population = replace_worst(self.population, child)


    def train(self) -> Chromosome:
        """
        Function to train the algorithm

        Returns:
            Chromosome - the best chromosome
        """

        for i in range(len(self.population.list)):
            self.population.list[i].calculate_fitness(self.inputs, self.outputs)

        for i in range(self.iterations):
            if i % self.epoch_feedback == 0:
                best_so_far = get_best(self.population)
                print(f"Best function: {best_so_far.gen}")
                print(f"Best fitness: {best_so_far.fitness}")
            self.__step()

        return get_best(self.population)
