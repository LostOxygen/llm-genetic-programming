"""Chromosome class implementation"""

import random
import torch
#import warnings
#warnings.filterwarnings("error")

class Chromosome:
    """
    Class for representing a chromosome
    """
    def __init__(self,
            terminal_set: list,
            funct_set: dict,
            depth: int,
            method: str = "full",
        ) -> None:
        """
        Constructor for Chromosome class

        Parameters:
            depth: int - tree depth
            method: str - method to generate the tree, default is full
            terminal_set: list - set of terminals
            funct_set: dict - set of functions

        Returns:
            None
        """
        self.depth = depth
        self.gen = [] # genotype, the tree of the chromosome
        self.terminal_set = terminal_set
        self.func_set = funct_set
        self.fitness = None
        if method == "grow":
            self.grow()
        elif method == "full":
            self.full()


    def full(self, level: int = 0) -> None:
        """
        Function to generate a tree in a full manner meaning that every node got two childs

        Parameters:
            level: int - current level in the tree

        Returns:
            None
        """
        if level == self.depth:
            self.gen.append(random.choice(self.terminal_set))
        else:
            val = random.choice(self.func_set[1] + self.func_set[2])
            if val in self.func_set[2]:
                self.gen.append(random.choice(self.func_set[2]))
                self.full(level + 1)
                self.full(level + 1)
            else:
                self.gen.append(random.choice(self.func_set[1]))
                self.full(level + 1)


    def grow(self, level = 0):
        """
        Function to generate a tree in a grow manner
        Every node may be a terminal or a function
        @return: None
        """
        if level == self.depth:
            self.gen.append(random.choice(self.terminal_set))
        else:
            if random.random() > 0.3:
                val = random.choice(self.func_set[2] + self.func_set[1])
                if val in self.func_set[2]:
                    self.gen.append(val)
                    self.grow(level + 1)
                    self.grow(level + 1)
                else:
                    self.gen.append(val)
                    self.grow(level + 1)
            else:
                val = random.choice(self.terminal_set)
                self.gen.append(val)


    def eval(self, func_input: list, curr_pos: int = 0):
        """
        Function to evaluate the current chromosome with a given input

        Parameters:
            func_input: List - function input [x0, x1... xn]
            curr_pos: int - current position in genotype

        Returns:
            Tuple - the value of the chromosome evaluated at the given input and the curr. position
        """
        if self.gen[curr_pos] in self.terminal_set:
            return func_input[int(self.gen[curr_pos][1:])], curr_pos
        elif self.gen[curr_pos] in self.func_set[2]:
            curr_pos_op = curr_pos
            left, curr_pos = self.eval(func_input, curr_pos + 1)
            right, curr_pos = self.eval(func_input, curr_pos + 1)
            if self.gen[curr_pos_op] == "+":
                return left + right, curr_pos
            elif self.gen[curr_pos_op] == "-":
                return left - right, curr_pos
            elif self.gen[curr_pos_op] == "*":
                return left * right, curr_pos
            elif self.gen[curr_pos_op] == "^":
                return left ** right, curr_pos
            elif self.gen[curr_pos_op] == "/":
                return left / right, curr_pos
        else:
            curr_pos_op = curr_pos
            left, curr_pos = self.eval(func_input, curr_pos + 1)
            if self.gen[curr_pos_op] == "sin":
                return torch.sin(left), curr_pos
            elif self.gen[curr_pos_op] == "cos":
                return torch.cos(left), curr_pos
            elif self.gen[curr_pos_op] == "ln":
                return torch.log(left), curr_pos
            elif self.gen[curr_pos_op] == "sqrt":
                return torch.sqrt(left), curr_pos
            elif self.gen[curr_pos_op] == "tg":
                return torch.tan(left), curr_pos
            elif self.gen[curr_pos_op] == "ctg":
                return 1/torch.tan(left), curr_pos
            elif self.gen[curr_pos_op] == "e":
                return torch.exp(left), curr_pos
            elif self.gen[curr_pos_op] == "tanh":
                return torch.tanh(left), curr_pos
            elif self.gen[curr_pos_op] == "abs":
                return abs(left), curr_pos


    def evaluate_arg(self, func_input: list):
        """
        Function to evaluate the current genotype to a given input

        Parameters:
            func_input: List - function input [x0, x1... xn]
        
        Returns:
            float - the value of self.gen evaluated at the given input
        """
        return self.eval(func_input)[0]


    def calculate_fitness(self, inputs: list, labels: list):
        """
        Function to claculate the fitness of a chromosome

        Parameters:
            inputs: List - inputs of the function we want to predict
            labels: List - outputs of the function we want to predict

        Returns:
            float - the chromosome's fitness (calculated based on MSE)
        """
        diff = 0
        for i in range(len(inputs)):
            try:
                diff += (self.eval(inputs[i])[0] - labels[i][0])**2
            except RuntimeWarning:
                self.gen = []
                if random.random() > 0.5:
                    self.grow()
                else:
                    self.full()
                self.calculate_fitness(inputs, labels)

        if len(inputs) == 0:
            return 1e9
        self.fitness = diff/(len(inputs))

        return self.fitness


    def __get_depth_aux(self, poz: int = 0):
        """
        Function to get the depth of a chromosome

        Parameters:

        @return: chromosome's depth, last pos
        """
        elem = self.gen[poz]

        if elem in self.func_set[2]:
            left, poz = self.__get_depth_aux(poz + 1)
            right, poz = self.__get_depth_aux(poz)

            return 1 + max(left, right), poz
        elif elem in self.func_set[1]:
            left, poz = self.__get_depth_aux(poz + 1)
            return left + 1, poz
        else:
            return 1, poz + 1


    def get_depth(self):
        """
        Function to get the depth of a chromosome
        @return: - chromosome's depth
        """
        return self.__get_depth_aux()[0] - 1
