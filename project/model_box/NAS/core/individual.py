from . import architecture_builder as builder
from copy import deepcopy
import numpy as np

class Individual:
    def __init__(self, max_layers, min_layers=3):
        self.architecture = builder.generate_random_architecture_code(
            max_layers=max_layers,
            min_layers=min_layers
        )
        self.chromosome = self.architecture2chromosome(
            input_architecture=self.architecture
        )
        self.parsed_layers = builder.parse_architecture_code(self.architecture)
        self.reset()
        self.id = np.uint64(np.random.randint(0, 2**64, dtype=np.uint64))

    def __str__(self):
        return f"Individual: {self.architecture} - objective values: {self.objectives}"

    def _reparse_layers(self):
        self.parsed_layers = builder.parse_architecture_code(
            self.chromosome2architecture(self.chromosome)
        )

    def reset(self):
        self.results = {}

        # raw evaluation metrics
        self.fps = 0
        self.miou = None
        self.metric = None   # optional alias for compatibility
        self.prediction_consistency = None
        self.std_dev = None

        # legacy field, no longer used for sorting
        self.fitness = None

        # NSGA-II objective vector:
        # minimize [-miou, -prediction_consistency, std_dev]
        self.objectives = None

        # NSGA-II metadata
        self.rank = None
        self.crowding_distance = 0.0

        # other metadata
        self.model_size = None
        self.model = None
        self.failed = False

    def set_objectives(self):
        if (
            self.miou is None
            or self.prediction_consistency is None
            or self.std_dev is None
        ):
            self.objectives = None
            return None

        self.objectives = [
            -float(self.miou),
            -float(self.prediction_consistency),
            float(self.std_dev),
        ]
        return self.objectives

    def architecture2chromosome(self, input_architecture):
        chromosome = input_architecture.split("E")
        if len(chromosome) >= 2 and chromosome[-1] == "" and chromosome[-2] == "":
            chromosome = chromosome[:-2]
        elif len(chromosome) >= 1 and chromosome[-1] == "":
            chromosome = chromosome[:-1]
        return chromosome

    def chromosome2architecture(self, input_chromosome):
        return "E".join(input_chromosome) + "EE"

    def copy(self):
        new_individual = Individual(max_layers=len(self.chromosome))
        new_individual.architecture = deepcopy(self.architecture)
        new_individual.chromosome = deepcopy(self.chromosome)
        new_individual.parsed_layers = deepcopy(self.parsed_layers)

        new_individual.results = deepcopy(self.results)

        new_individual.miou = self.miou
        new_individual.metric = self.metric
        new_individual.prediction_consistency = self.prediction_consistency
        new_individual.std_dev = self.std_dev

        new_individual.fitness = self.fitness
        new_individual.objectives = deepcopy(self.objectives)

        new_individual.rank = self.rank
        new_individual.crowding_distance = self.crowding_distance

        new_individual.model_size = self.model_size
        new_individual.failed = self.failed

        if self.model is not None:
            new_individual.model = deepcopy(self.model)

        return new_individual

    def set_trained_model(self, model):
        self.model = model