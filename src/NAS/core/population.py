import pandas as pd
import numpy as np
import pickle
from copy import deepcopy
import tqdm, os
import json
import subprocess
import tempfile
import shutil

import time

import gc

from ..blocks.heads import MultiInputClassifier
from .individual import Individual 
from .generic_unet import GenericUNetNetwork
from ..opt.evo import single_point_crossover, gene_mutation
from .generic_lightning_module import GenericLightningSegmentationNetwork, GenericLightningNetwork



from src.perturbation_methods.perturbation_methods import *
from src.NAS.train.evaluation_functions import compute_miou, compute_prediction_consistency
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.operators.survival.rank_and_crowding.metrics import get_crowding_function
from pymoo.operators.survival.rank_and_crowding import RankAndCrowding
from pymoo.core.population import Population as PymooPopulation
from pymoo.core.individual import Individual as PymooIndividual
from pymoo.core.problem import Problem

import logging 

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.multiprocessing as mp
from pytorch_lightning.callbacks import EarlyStopping

from IPython.display import clear_output


# Update config_path to use the directory of the current file
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')

#mp.set_start_method('fork', force=True)
import torch.multiprocessing as mp





try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

def update_config_path(config_path, model_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    config['model_filepath'] = model_path
    with open(config_path, 'w') as f:
        json.dump(config, f)

class DummyProblem(Problem):
    def __init__(self, n_obj):
        super().__init__(n_var=0, n_obj=n_obj, n_ieq_constr=0, n_eq_constr=0)

    def _evaluate(self, X, out, *args, **kwargs):
        pass


def _update_confmat(preds, targets, num_classes):
    """
    preds:   [B, H, W] long
    targets: [B, H, W] long
    """
    valid = (targets >= 0) & (targets < num_classes)

    inds = num_classes * targets[valid] + preds[valid]
    confmat = torch.bincount(
        inds,
        minlength=num_classes * num_classes
    ).reshape(num_classes, num_classes)

    return confmat


def _miou_from_confmat(confmat):
    confmat = confmat.float()

    intersection = torch.diag(confmat)
    union = (
        confmat.sum(dim=1) +  # target pixels per class
        confmat.sum(dim=0) -  # predicted pixels per class
        intersection
    )

    valid = union > 0
    if valid.sum() == 0:
        return torch.tensor(0.0, device=confmat.device)

    return (intersection[valid] / union[valid]).mean()

class Population:
    def __init__(self, n_individuals, max_layers, dm, max_parameters=100_000, save_directory=None, run_name="", perturbation="clean", strength=0):
        """
        Initialize a new population for the evolutionary neural architecture search.
        
        Parameters:
            n_individuals (int): Number of individuals in the population
            max_layers (int): Maximum number of layers in an individual's architecture
            dm (object): Data module for model creation and evaluation
            max_parameters (int, optional): Maximum number of parameters allowed in a model. Defaults to 100,000.
            save_directory (str, optional): Directory to save models and checkpoints. Defaults to "./models_traced".
        
        Raises:
            ValueError: If input parameters are invalid (negative values, none data module)
        """
        # Validate input parameters
        if not isinstance(n_individuals, int) or n_individuals <= 0:
            raise ValueError(f"n_individuals must be a positive integer, got {n_individuals}")
        if not isinstance(max_layers, int) or max_layers <= 0:
            raise ValueError(f"max_layers must be a positive integer, got {max_layers}")
        if dm is None:
            raise ValueError("Data module (dm) cannot be None")
        if not isinstance(max_parameters, int) or max_parameters <= 0:
            raise ValueError(f"max_parameters must be a positive integer, got {max_parameters}")
        
        # Data and model parameters
        self.dm = dm  # Data module for model creation
        self.n_individuals = n_individuals
        self.max_layers = max_layers
        self.max_parameters = max_parameters
        
        # State tracking
        self.generation = 0
        self.population = []  # Initialize empty population
        self.df = None  # Will hold population stats as DataFrame
        self.run_name = run_name
        # File storage
        self.save_directory = save_directory or "./models_traced"
        # Create directories if they don't exist
        os.makedirs(os.path.join(self.save_directory, "src"), exist_ok=True)
        os.makedirs(os.path.join(self.save_directory, "backups"), exist_ok=True)
        
        # Hardware
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = self.setup_logger(enable_file_logging=False)
        
        self.perturbation = perturbation
        self.strength = strength

        self.logger.info(f"Initialized population with {n_individuals} individuals, "
                         f"max_layers={max_layers}, max_parameters={max_parameters}, "
                         f"device={self.device}")
        
    @staticmethod
    def setup_logger(log_file='./logs/population.log', log_level=logging.INFO, enable_file_logging=True):
        logger = logging.getLogger(__name__)
        logger.setLevel(log_level)
        logger.propagate = False

        # Clear old handlers to prevent duplicate logging
        logger.handlers.clear()

        if not enable_file_logging:
            logger.addHandler(logging.NullHandler())
            return logger

        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setLevel(log_level)

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        return logger
        
    def initial_poll(self):
        """
        Generate the initial population of individuals.    
        """
        
        self.population = self.create_population()
        self._checkpoint()


    def create_random_individual(self, max_attempts=5):
        """
        Create a random individual with a random number of layers.
        
        This function attempts to create a valid random individual with
        proper error handling and retry logic to ensure robustness.
        
        Parameters:
            max_attempts (int): Maximum number of attempts to create a valid individual.
                               Defaults to 5.
        
        Returns:
            Individual: A valid random individual.
            
        Raises:
            RuntimeError: If unable to create a valid individual after max_attempts.
        """
        for attempt in range(max_attempts):
            try:
                individual = Individual(max_layers=self.max_layers)
                
                # Basic validation that the individual was created properly
                if not hasattr(individual, 'parsed_layers') or not individual.parsed_layers:
                    self.logger.warning(f"Created individual has invalid parsed_layers (attempt {attempt+1}/{max_attempts})")
                    continue
                    
                self.logger.debug(f"Successfully created random individual with {len(individual.parsed_layers)} layers")
                return individual
                
            except Exception as e:
                self.logger.warning(f"Failed to create random individual (attempt {attempt+1}/{max_attempts}): {str(e)}")
        
        # If we reach here, all attempts failed
        error_msg = f"Failed to create valid random individual after {max_attempts} attempts"
        self.logger.error(error_msg)
        raise RuntimeError(error_msg)
    

    def _sort_population(self):
        """
        Sort the population by fitness in descending order.
        
        This method:
        1. Validates the population exists and is not empty
        2. Handles individuals with missing or invalid fitness values
        3. Provides comprehensive error handling
        4. Logs sorting operations for debugging
        
        Returns:
            list: Sorted population by fitness (descending order)
        """
        # Check if population exists and is not empty
        if not hasattr(self, 'population') or not self.population:
            self.logger.warning("Cannot sort population: population is empty or not initialized")
            return []
        
        try:
            # Filter out individuals with invalid fitness values
            valid_individuals = []
            invalid_count = 0
            
            for individual in self.population:
                # Check if the individual has a fitness attribute and it's a valid value
                if (hasattr(individual, 'fitness') and 
                    individual.fitness is not None and 
                    not np.isnan(individual.fitness)):
                    valid_individuals.append(individual)
                else:
                    invalid_count += 1
            
            if invalid_count > 0:
                self.logger.warning(f"Found {invalid_count} individuals with invalid fitness values")
            
            if not valid_individuals:
                self.logger.error("No individuals with valid fitness values found!")
                return self.population  # Return unsorted population as fallback
            
            # Sort the valid individuals
            self.logger.debug(f"Sorting {len(valid_individuals)} individuals by fitness")
            sorted_population = sorted(valid_individuals, key=lambda ind: ind.fitness, reverse=True)
            
            # Update the population with sorted individuals
            self.population = sorted_population
            
            # Log the top fitness values for debugging
            if sorted_population:
                top_fitness = [ind.fitness for ind in sorted_population[:min(3, len(sorted_population))]]
                self.logger.info(f"Top fitness values after sorting: {top_fitness}")
            
            # Checkpoint the sorted population (with error handling)
            try:
                self._checkpoint()
            except Exception as e:
                print(f"Error during checkpointing after sorting: {str(e)}")
                self.logger.error(f"Failed to checkpoint after sorting: {str(e)}")
            
            return sorted_population
            
        except Exception as e:
            print(f"Population sorting failed with error: {str(e)}")
            self.logger.error(f"Population sorting failed with error: {str(e)}")
            return self.population  # Return unsorted population as fallback
        

    def _checkpoint(self):
        """
        Save the current population state to disk, including dataframes and serialized models.
        
        This implementation includes:
        - Validation of population state before saving
        - Comprehensive error handling for each saving step
        - Backup of previous checkpoints
        - Detailed logging
        """
        if not hasattr(self, 'population') or not self.population:
            self.logger.error("Cannot checkpoint: population is empty or not initialized")
            return False
        
        try:
            # Create save directory if it doesn't exist
            os.makedirs(self.save_directory, exist_ok=True)
            
            # Create backup directory for current generation
            backup_dir = os.path.join(self.save_directory, f"backups/gen_{self.generation}")
            os.makedirs(backup_dir, exist_ok=True)
            
            # Backup previous files if they exist
            for file_type in ["population", "df_population"]:
                src_path = f'{self.save_directory}/src/{file_type}_{self.generation}.pkl'
                if os.path.exists(src_path):
                    backup_path = f'{backup_dir}/{file_type}_{self.generation}_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.pkl'
                    try:
                        import shutil
                        shutil.copy2(src_path, backup_path)
                        self.logger.debug(f"Backed up {src_path} to {backup_path}")
                    except Exception as e:
                        self.logger.warning(f"Failed to backup {src_path}: {e}")
            
            # Update dataframe with current population stats
            try:
                self._update_df()
                self.logger.debug("Updated population dataframe")
            except Exception as e:
                self.logger.error(f"Failed to update dataframe: {e}")
                return False
            
            # Save population and dataframe
            save_success = True
            
            try:
                self.save_population()
            except Exception as e:
                self.logger.error(f"Failed to save population: {e}")
                save_success = False
                
            try:
                self.save_dataframe()
            except Exception as e:
                self.logger.error(f"Failed to save dataframe: {e}")
                save_success = False
                
            if save_success:
                self.logger.info(f"Successfully checkpointed population at generation {self.generation}")
                return True
            else:
                self.logger.warning(f"Checkpoint at generation {self.generation} was incomplete")
                return False
                
        except Exception as e:
            self.logger.error(f"Checkpoint failed with error: {e}")
            return False
    
    
    def check_individual(self, individual):
        """
        Validate if an individual can be built into a functional model with acceptable parameters.
        
        This method:
        1. Validates the input individual object
        2. Attempts to build a model from the individual's genetic representation
        3. Evaluates the model's parameter count
        4. Ensures the model meets size constraints
        5. Updates the individual with its model_size
        
        Parameters:
            individual (Individual): The individual to check
            
        Returns:
            bool: True if the individual is valid, False otherwise
        """
        if individual is None:
            self.logger.error("Cannot check individual: received None")
            return False
            
        if not hasattr(individual, 'parsed_layers') or not individual.parsed_layers:
            self.logger.error(f"Individual is missing parsed_layers attribute or it's empty")
            return False
        
        try:
            # Attempt to build the model
            self.logger.debug(f"Building model from individual with {len(individual.parsed_layers)} layers")
            model_representation, is_valid = self.build_model(individual.parsed_layers)
            
            if not is_valid:
                self.logger.warning(f"Model building failed for individual: build_model returned is_valid=False")
                return False
                
            # Evaluate the model's parameter count
            try:
                modelSize = self.evaluate_parameters(model_representation)
                individual.model_size = modelSize
                self.logger.debug(f"Model size: {modelSize} parameters")
            except Exception as e:
                self.logger.error(f"Failed to evaluate model parameters: {e}")
                return False
            
            # Validate the model size
            if modelSize <= 0:
                self.logger.warning(f"Invalid model size: {modelSize} (must be positive)")
                return False
                
            if modelSize >= self.max_parameters:
                self.logger.warning(f"Model too large: {modelSize} parameters (max: {self.max_parameters})")
                return False
                
            if modelSize is None:
                self.logger.warning("Model size is None")
                return False
            
            # All checks passed
            self.logger.debug(f"Individual passed all validation checks, model size: {modelSize}")
            return True
            
        except AssertionError as e:
            self.logger.warning(f"Assertion failed during individual check: {e}")
            return False
        except ValueError as e:
            self.logger.warning(f"Value error during individual check: {e}")
            return False
        except RuntimeError as e:
            self.logger.warning(f"Runtime error during individual check: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error checking individual: {str(e)}")
            return False


    def create_population(self, max_attempts=200, timeout_seconds=300):
        """
        Create a population of unique, valid individuals.
    
        This function generates random individuals and checks if they're valid using check_individual.
        It includes comprehensive error handling, duplicate removal, and recovery mechanisms.
    
        Parameters:
            max_attempts (int): Maximum number of attempts to create valid individuals. Default: 200
            timeout_seconds (int): Maximum time in seconds before giving up. Default: 300 (5 minutes)
    
        Returns:
            list: A list of unique, valid individuals.
            
        Raises:
            RuntimeError: If unable to generate a complete population after max_attempts
        """
        import time
        start_time = time.time()
        population = []
        attempts = 0
        failed_attempts = 0
        additional_attempts = 0  # Initialize here to avoid UnboundLocalError
        
        # Create progress bar for initial population generation
        with tqdm.tqdm(total=self.n_individuals, desc="Generating Population") as pbar:
            while len(population) < self.n_individuals:
                # Check timeout
                if time.time() - start_time > timeout_seconds:
                    self.logger.warning(f"Population generation timed out after {timeout_seconds} seconds. "
                                       f"Created {len(population)}/{self.n_individuals} individuals.")
                    break
                    
                # Check max attempts
                if attempts >= max_attempts:
                    self.logger.warning(f"Reached maximum attempts ({max_attempts}) for population generation. "
                                       f"Created {len(population)}/{self.n_individuals} individuals.")
                    break
                    
                attempts += 1
                
                try:
                    # Create a random individual
                    candidate = self.create_random_individual()
                    
                    # Check if the individual is valid
                    if self.check_individual(candidate):
                        population.append(candidate)
                        pbar.update(1)  # Update progress bar
                        self.logger.debug(f"Added individual {len(population)}/{self.n_individuals} "
                                         f"(attempt {attempts}, failed: {failed_attempts})")
                    else:
                        failed_attempts += 1
                except Exception as e:
                    failed_attempts += 1
                    self.logger.warning(f"Failed to create individual on attempt {attempts}: {e}")
                    
                # Periodically log progress
                if attempts % 10 == 0:
                    self.logger.info(f"Population generation: {len(population)}/{self.n_individuals} created "
                                    f"(attempts: {attempts}, failed: {failed_attempts})")
        
        # Handle duplicates and ensure we have enough individuals
        original_count = len(population)
        self.logger.info(f"Initial population created with {original_count} individuals, removing duplicates...")
        
        # First round of duplicate removal
        population = self.remove_duplicates(population)
        
        # If removing duplicates reduced the population, attempt to fill it back up
        if len(population) < self.n_individuals:
            self.logger.warning(f"Population size after duplicate removal: {len(population)}/{self.n_individuals}")
            self.logger.info(f"Attempting to generate additional {self.n_individuals - len(population)} unique individuals")
            
            # Create a separate progress bar for filling the missing individuals
            with tqdm.tqdm(total=self.n_individuals - len(population), desc="Filling Missing") as pbar:
                additional_attempts = 0
                fill_start_time = time.time()
                
                while len(population) < self.n_individuals:
                    # Check timeout and max attempts
                    if time.time() - fill_start_time > timeout_seconds / 2:  # Allow half the original timeout
                        self.logger.warning("Timed out while trying to fill population after duplicate removal")
                        break
                        
                    if additional_attempts >= max_attempts / 2:  # Allow half the original max attempts
                        self.logger.warning("Reached maximum attempts while trying to fill population after duplicate removal")
                        break
                        
                    additional_attempts += 1
                    
                    try:
                        # Check current architectures to avoid creating duplicates
                        existing_archs = set(getattr(ind, 'architecture', str(ind.parsed_layers)) for ind in population)
                        
                        # Create a new individual
                        candidate = self.create_random_individual()
                        
                        # Check if it's valid and not a duplicate
                        if self.check_individual(candidate):
                            new_arch = getattr(candidate, 'architecture', str(candidate.parsed_layers))
                            if new_arch not in existing_archs:
                                population.append(candidate)
                                existing_archs.add(new_arch)
                                pbar.update(1)
                                self.logger.debug(f"Added missing individual {len(population)}/{self.n_individuals}")
                    except Exception as e:
                        self.logger.warning(f"Failed while filling population: {e}")
        
        # Final duplicate check and warning
        final_unique_count = len(set(getattr(ind, 'architecture', str(ind.parsed_layers)) for ind in population))
        if final_unique_count < len(population):
            self.logger.warning(f"Final population still contains duplicates: "
                              f"{len(population) - final_unique_count} duplicates detected")
        
        # Log final statistics
        self.logger.info(f"Population generation completed. Created {len(population)}/{self.n_individuals} individuals "
                       f"in {time.time() - start_time:.1f} seconds "
                       f"(attempts: {attempts + additional_attempts}, success rate: "
                       f"{len(population)/(attempts + additional_attempts):.1%})")
        
        # If we couldn't create enough individuals, log an error
        if len(population) < self.n_individuals:
            self.logger.error(f"Unable to create required population size. Created only "
                             f"{len(population)}/{self.n_individuals} individuals.")
            if len(population) < self.n_individuals * 0.5:  # Less than 50% of required individuals
                raise RuntimeError(f"Failed to create a viable population. Only generated "
                                 f"{len(population)}/{self.n_individuals} individuals.")
        
        return population


    # def elite_models(self, k_best=1):
    #     """
    #     Retrieve the top k_best elite models from the current population based on fitness.

    #     The population is sorted in descending order based on the fitness attribute of each individual.
    #     This function then returns deep copies of the top k_best individuals to ensure that the
    #     original models remain immutable during further operations.

    #     Parameters:
    #         k_best (int): The number of top-performing individuals to retrieve. Defaults to 1.

    #     Returns:
    #         list: A list containing deep copies of the elite individuals.
    #     """
    #         # Filter out individuals with invalid fitness values
    #     valid_individuals = [ind for ind in self.population if hasattr(ind, 'fitness') 
    #                         and ind.fitness is not None 
    #                         and not np.isnan(ind.fitness)]
        
    #     if not valid_individuals:
    #         self.logger.warning("No valid individuals with fitness values found!")
    #         return []
    #     sorted_pop = self._sort_population()
    #     # Ensure we don't request more models than are available
    #     k_best = min(k_best, len(sorted_pop))
    #     # Create deep copies of the top models
    #     topModels = [deepcopy(sorted_pop[i]) for i in range(k_best)]
    #     # Log the fitness of selected models for debugging
    #     for i, model in enumerate(topModels):
    #         self.logger.info(f"Selected elite model for next generation. Idx {i} with fitness: {model.fitness}")
    #     return topModels

    def elite_models(self, k_best=1):
        valid = [
        ind for ind in self.population
        if getattr(ind, "objectives", None) is not None
        and np.all(np.isfinite(ind.objectives))
        ]

        if not valid:
            return []

        inds = [
            PymooIndividual(
                F=np.asarray(ind.objectives, dtype=float),
                CV=0.0,
                idx=i
            )
            for i, ind in enumerate(valid)
        ]

        pymoo_pop = PymooPopulation.create(*inds)

        problem = DummyProblem(n_obj=len(valid[0].objectives))
        survival = RankAndCrowding(crowding_func="cd")

        # run survival on the full population so rank/crowding get assigned
        _ = survival.do(
            problem=problem,
            pop=pymoo_pop,
            n_survive=len(pymoo_pop)
        )

        annotated = []
        for p in pymoo_pop:
            idx = p.get("idx")
            ind = valid[idx]
            rank = p.get("rank")
            crowding = p.get("crowding")

            annotated.append((ind, rank, crowding))

        # NSGA-II ordering: rank ascending, crowding descending
        annotated.sort(key=lambda x: (x[1], -x[2]))


        return [deepcopy(x[0]) for x in annotated]

    def evolve(self, mating_pool_cutoff=0.5, mutation_probability=0.85, k_best=1, n_random=3):
        """
        Generates a new population ensuring that the total number of individuals equals pop.n_individuals.
        
        Parameters:
            pop                  : List or collection of individuals. Assumed to have attributes: 
                                .n_individuals and .generation.
            mating_pool_cutoff   : Fraction determining the size of the mating pool (top percent of individuals).
            mutation_probability : The probability to use during mutation.
            k_best               : The number of best individuals from the current population to retain.
        
        Returns:
            new_population: A list representing the new generation of individuals.
            
        Note:
            Assumes that helper functions single_point_crossover(), mutation(), and create_random_individual() exist.
        """
        new_population = []
        self.generation += 1

        sorted_pop = self.elite_models()

        self.topModels = sorted_pop[:k_best]


        mating_pool_size = max(1, int(np.floor(mating_pool_cutoff * self.n_individuals)))
        mating_pool = sorted_pop[:mating_pool_size].copy()

        assert len(mating_pool) > 0, "Mating pool is empty."

        # keep elites
        new_population.extend(self.topModels)
    
        # Generate offspring until reaching the desired population size
        while len(new_population) < self.n_individuals - n_random - k_best:
            try:
                parent1 = np.random.choice(mating_pool)
                parent2 = np.random.choice(mating_pool)
                assert parent1.parsed_layers != parent2.parsed_layers, "Parents are the same individual."
            except Exception as e:
                self.logger.error(f"Error selecting parents: {e}")
                continue
            
            # a) Crossover:
            children = single_point_crossover([parent1, parent2])
            # b) Mutation:
            mutated_children = gene_mutation(children, mutation_probability)
            # c) Random choice of one of the mutated children
            for kid in mutated_children:
                kid.reset()
                if self.check_individual(kid):
                    new_population.append(kid)
                else:
                    pass


        # 3. Add random individuals to the new population
        while len(new_population) < self.n_individuals - k_best:
            try:
                individual = self.create_random_individual()
                model_representation, is_valid = self.build_model(individual.parsed_layers)
                if is_valid:
                    individual.model_size = int(self.evaluate_parameters(model_representation))
                    assert individual.model_size > 0, f"Model size is {individual.model_size}"
                    assert individual.model_size < self.max_parameters, f"Model size is {individual.model_size}"
                    assert individual.model_size is not None, f"Model size is None"
                    new_population.append(individual)
            except Exception as e:
                self.logger.error(f"Error encountered when evolving population: {e}")
                continue
        
        
        # 4. Add the best individuals from the previous generation
        new_population.extend(self.topModels)
        # Give new population models each a new id
        for individual in new_population:
            individual.id = np.uint64(np.random.randint(0, 2**64, dtype=np.uint64))

        assert len(new_population) == self.n_individuals, f"Population size is {len(new_population)}, expected {self.n_individuals}"
        self.population = new_population
        self._checkpoint()


    def remove_duplicates(self, population):
        """
        Remove duplicates from the given population by replacing duplicates with newly generated unique individuals.

        Parameters:
            population (list): A list of individuals in the population.

        Returns:
            list: The updated population with duplicates removed.
        """
        unique_architectures = set()
        updated_population = []

        for individual in population:
            # Use the 'architecture' attribute if available, otherwise fallback to a default representation.
            arch = getattr(individual, 'architecture', None)
            if arch is None:
                # If no architecture attribute, use parsed_layers as unique identifier.
                arch = str(individual.parsed_layers)

            if arch not in unique_architectures:
                unique_architectures.add(arch)
                updated_population.append(individual)
            else:
                # Try to generate a unique individual up to 50 times
                for _ in range(50):
                    new_individual = Individual(max_layers=self.max_layers)
                    new_arch = getattr(new_individual, 'architecture', None)
                    if new_arch is None:
                        new_arch = str(new_individual.parsed_layers)

                    if new_arch not in unique_architectures:
                        unique_architectures.add(new_arch)
                        updated_population.append(new_individual)
                        break
                else:
                    # After 50 attempts, keep the original duplicate as a fallback.
                    updated_population.append(individual)
        return updated_population
        
    
    def build_model(self, parsed_layers, task="segmentation"):
        """
        Build a model based on the provided parsed layers.

        This function creates an encoder using the parsed layers and constructs a model by combining
        the encoder with a head layer via the ModelConstructor. The constructed model is built to
        process inputs defined by the data module (dm).

        Parameters:
            parsed_layers: The parsed architecture configuration used by the encoder to build the network.

        Returns:
            A PyTorch model constructed with the encoder and head layer.
        """
        
        def shape_tracer(self, encoder):
            """
            Traces the output shapes of a given encoder model when provided with a dummy input.
            Args:
                encoder (torch.nn.Module): The encoder model whose output shapes are to be traced.
            Returns:
                list[tuple]: A list of tuples representing the shapes of the encoder's outputs 
                     (excluding the batch dimension). If the encoder outputs a single tensor, 
                     the list will contain one tuple. If the encoder outputs multiple tensors 
                     (e.g., a list or tuple of tensors), the list will contain a tuple for each output.
            """
            
            dummy_input = torch.randn(1, *self.dm.input_shape).to(self.device)
            with torch.no_grad():
                output = encoder(dummy_input)
            shapes = []
            if isinstance(output, (list, tuple)):
                for o in output:
                    shape_without_batch = tuple(o.shape[1:])
                    shapes.append(shape_without_batch)
            else:
                shape_without_batch = tuple(output.shape[1:])
                shapes.append(shape_without_batch)
            self.logger.debug(f"Shape tracer output: {shapes}")
            return shapes
        
        self.task = task
        
        if task == "segmentation":
            model = GenericUNetNetwork(parsed_layers,
                    input_channels=self.dm.input_shape[0], 
                    input_height=self.dm.input_shape[1], 
                    input_width=self.dm.input_shape[2], 
                    num_classes=self.dm.num_classes,
                    encoder_only=False,
            )
            valid = True
        elif task == "classification": 
            encoder = GenericUNetNetwork(parsed_layers,
                    input_channels=self.dm.input_shape[0], 
                    input_height=self.dm.input_shape[1], 
                    input_width=self.dm.input_shape[2], 
                    num_classes=self.dm.num_classes,
                    encoder_only=True,
            )
            valid = True
                
            head = MultiInputClassifier(shape_tracer(self, encoder.to(self.device)), num_classes=self.dm.num_classes)
            head = head.to(self.device)
            model = nn.Sequential(encoder, head)
            
        else:
            raise ValueError(f"Task {task} not supported.")
        
        return model, valid
    
    
    def evaluate_parameters(self, model):
        """
        Calculate the total number of parameters of the given model.

        Parameters:
            model (torch.nn.Module): The PyTorch model.

        Returns:
            int: The total number of parameters.
        """
        num_params = sum(p.numel() for p in model.parameters())
        return num_params
    
    
    def _update_df(self):
        """
        Create a DataFrame from the population.

        Returns:
            pd.DataFrame: A DataFrame containing the population.
        """
        columns = ["Generation", "Layers", "Fitness", "Metric", "FPS", "Params"]
        data = []
        for individual in self.population:
            generation = self.generation
            parsed_layers = individual.parsed_layers
            metric = individual.metric
            fps = individual.fps
            fitness = individual.fitness
            model_size = individual.model_size
            data.append([generation, parsed_layers, fitness, metric, fps, model_size])
        
        #df = pd.DataFrame(data, columns=columns).sort_values(by="Fitness", ascending=False)
        #df.reset_index(drop=True, inplace=True)
        # DO NOT SORT OR RESET INDEX — we want df index to match self.population[idx]
        df = pd.DataFrame(data, columns=columns)
        
        self.df = df
    
    
    def save_dataframe(self):
        """
        Save the DataFrame containing the population statistics to a pickle file.

        The DataFrame is saved at a path that includes the current generation number.
        In case of an error during saving, the exception details are printed.

        Returns:
            None
        """
        path = f'{self.save_directory}/src/df_population_{self.generation}.pkl'
        try:
            self.df.to_pickle(path)
            self.logger.info(f"DataFrame saved to {path}")
        except Exception as e:
            self.logger.error(f"Error saving DataFrame to {path}: {e}")
    
    
    def load_dataframe(self, generation):
        path = f'./models_traced/src/df_population_{generation}.pkl'
        try:
            df = pd.read_pickle(path)
            return df
        except Exception as e:
            self.logger.error(f"Error loading DataFrame from {path}: {e}")
            return None
    
    
    def save_population(self):
        path = f'./models_traced/src/population_{self.generation}.pkl'
        try:
            with open(path, 'wb') as f:
                pickle.dump(self.population, f)
            self.logger.info(f"Population saved to {path}")
        except Exception as e:
            self.logger.error(f"Error saving population to {path}: {e}")
    
    
    def load_population(self, generation):
        path = f'./models_traced/src/population_{generation}.pkl'
        try:
            with open(path, 'rb') as f:
                population = pickle.load(f)
            return population
        except Exception as e:
            self.logger.error(f"Error loading population from {path}: {e}")
            return None

    def train_individual(self, idx, task, epochs=20, lr=1e-3, batch_size=None):
        """
        Train the individual using the data module and the specified number of epochs and learning rate.

        Parameters:
            individual (Individual): The individual to train.
            epochs (int): The number of epochs to train the individual. Defaults to 20.
            lr (float): The learning rate to use during training. Defaults to 1e-3.

        Returns:
            None
        """
        individual = self.population[idx]
        
        
        model, _ = self.build_model(individual.parsed_layers, task=task)
        if task == "segmentation":
            LM = GenericLightningSegmentationNetwork(
                model=model,
                learning_rate=lr,
            )
        
        elif task == "classification":
            LM = GenericLightningNetwork(
                model=model,
                learning_rate=lr,
                num_classes=self.dm.num_classes,
            )
        else:
            raise ValueError(f"Task {task} not supported.")


        early_stop_callback = EarlyStopping(
                            monitor="val_loss",     # or "val_iou" or any metric you're logging
                            mode="min",             # "min" if loss, "max" if accuracy or IoU
                            patience=3,             # number of epochs with no improvement
                            verbose=False)

        
        # Create a PyTorch Lightning trainer
        trainer = pl.Trainer(
                            #strategy="ddp_notebook",
                            accelerator="gpu",
                            devices=1,
                            max_epochs=epochs,
                            callbacks=[early_stop_callback]   
                            )
        # Set the batch size if specified
        if batch_size is not None:
            self.dm.batch_size = batch_size
        # Train the lightning model
        print("Strategy in use:", trainer.strategy)
        trainer.fit(LM, self.dm)
        
        individual.id = np.uint64(np.random.randint(0, 2**64, dtype=np.uint64))

        results = self.evaluate_individual(LM, task=task)
        
        self.log_results(individual, results)

        print(individual.id, results)
        
        self.results = results

        individual.miou = results["miou_clean"]
        individual.metric = results["miou_clean"]
        individual.prediction_consistency = results["prediction_consistency"]
        individual.process_time = results["process_time"]
        individual.set_objectives()

        self._checkpoint()
        print('checkpointed')
        self.save_model(LM, idx)


        LM.model.cpu()
        del LM
        
    def log_results(self, individual, results):
        log_entry = {
            "generation": self.generation,
            "id": int(individual.id),
            "results": results
        }

        with open(f"{self.run_name}_evolution_log.jsonl", "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    def perturb_batch(self, x):
        """
        Perturb a batch x using self.perturbation and self.strength.

        Supported perturbations:
            "noise"        -> SNR degradation, strength = snr_factor
            "blur"         -> MTF@Nyquist, strength = mtf_nyquist
            "misalignment" -> max band shift in pixels, strength = max_shift_px
            "haze"         -> transmission t, strength = t
            "brightness"   -> multiplicative factor alpha, strength = alpha

        Input:
            x: torch.Tensor, shape (B, C, H, W)

        Output:
            torch.Tensor, same shape/device/dtype as x
        """
        perturbation = self.perturbation
        strength = self.strength

        if perturbation is None or perturbation == "none":
            return x

        device = x.device
        dtype = x.dtype

        x_np = x.detach().cpu().numpy().astype(np.float32)
        x_out = np.empty_like(x_np, dtype=np.float32)

        for i in range(x_np.shape[0]):
            xi = x_np[i]

            if perturbation == "noise":
                xi_out = perturb_snr(
                    xi,
                    snr_factor=float(strength),
                    use_official_lref=True,
                    seed=None,
                )

            elif perturbation == "blur":
                xi_out = perturb_mtf(
                    xi,
                    mtf_nyquist=float(strength),
                )

            elif perturbation == "misalignment":
               # reference_band = 0 if xi.shape[0] == 8 else None TODO has to be red band - fix this in perturbation function code
                xi_out = perturb_band_misalignment(
                    xi,
                    max_shift_px=float(strength),
                    #reference_band=reference_band,
                    seed=None,
                )

            elif perturbation == "haze":
                xi_out = perturb_haze(
                    xi,
                    t=float(strength),
                    atmospheric_light="p95",
                )

            elif perturbation == "brightness":
                xi_out = perturb_brightness(
                    xi,
                    alpha=float(strength),
                )

            else:
                raise ValueError(
                    f"Unknown perturbation {perturbation!r}. "
                    "Expected one of: 'noise', 'blur', 'misalignment', "
                    "'haze', 'brightness', 'none'."
                )

            x_out[i] = xi_out.astype(np.float32)

        return torch.from_numpy(x_out).to(device=device, dtype=dtype)

    def evaluate_individual(self, LM, task=None):
        print("Evaluating individual...")

        LM.eval()
        LM.to(self.device)

        num_classes = 4

        clean_confmat = torch.zeros(
            num_classes, num_classes, device=self.device, dtype=torch.long
        )
        pert_confmat = torch.zeros(
            num_classes, num_classes, device=self.device, dtype=torch.long
        )

        consistency_correct = torch.zeros((), device=self.device, dtype=torch.long)
        consistency_total = torch.zeros((), device=self.device, dtype=torch.long)

        self.dm.setup(stage="test")
        test_loader = self.dm.test_dataloader()

        process_time = 0.0

        with torch.inference_mode():
            for batch_idx, batch in enumerate(test_loader):
                # raw/un-normalized batch
                x, y = batch

                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                y_indices = torch.argmax(y, dim=1)

                # Always create both versions from raw x
                if self.perturbation != "clean":
                    x_perturbed = self.perturb_batch(x)
                else:
                    x_perturbed = x.clone()

                # Normalize after perturbation
                x = self.dm.test_dataset.normalize_batch(x)
                x_perturbed = self.dm.test_dataset.normalize_batch(x_perturbed)

                # Now both are normalized
                x_all = torch.cat([x, x_perturbed], dim=0)

                if self.device.type == "cuda":
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)

                    start.record()
                    logits_all = LM(x_all)
                    end.record()

                    torch.cuda.synchronize()
                    process_time += start.elapsed_time(end) / 1000.0
                else:
                    t1 = time.time()
                    logits_all = LM(x_all)
                    t2 = time.time()
                    process_time += t2 - t1

                logits, logits_pert = torch.chunk(logits_all, chunks=2, dim=0)

                preds = torch.argmax(logits, dim=1)
                preds_pert = torch.argmax(logits_pert, dim=1)

                clean_confmat += _update_confmat(preds, y_indices, num_classes)
                pert_confmat += _update_confmat(preds_pert, y_indices, num_classes)

                consistency_correct += (preds == preds_pert).sum()
                consistency_total += preds.numel()

        miou_clean = _miou_from_confmat(clean_confmat)
        miou_perturbed = _miou_from_confmat(pert_confmat)
        prediction_consistency = consistency_correct.float() / consistency_total.float()

        return {
            "miou_clean": float(miou_clean.detach().cpu()),
            "miou_perturbed": float(miou_perturbed.detach().cpu()),
            "prediction_consistency": float(prediction_consistency.detach().cpu()),
            "process_time": float(process_time),
        }

    def train_generation(self, task='classification', lr=0.001, epochs=4, batch_size=32):
        """
        Train all individuals in the current generation that have not been trained yet.

        Parameters:
            task (str): The task type ('classification' or 'segmentation').
            lr (float): Learning rate for training.
            epochs (int): Number of epochs for training.
            batch_size (int): Batch size for training.

        Returns:
            None
        """
        for idx in range(len(self)):
            # if 'Fitness' in self.df.columns and not pd.isna(self.df.loc[idx, 'Fitness']) and self.df.loc[idx, 'Fitness'] != 0:
            #     print(f"Skipping individual {idx}/{len(self)} as it has already been trained")
            #     continue

            # print(f"Training individual {idx}/{len(self)}")
            self.train_individual(idx=idx, task=task, lr=lr, epochs=epochs, batch_size=batch_size)
            #clear_output(wait=True)

    def save_model(self, LM, idx):
        individual = self.population[idx]
        gen = self.generation
        ind_id = int(individual.id)
        #perturb = self.perturbation if self.perturbation not in [None, ""] else "clean"

        save_dir = os.path.join(
            self.save_directory,
            f"generation_{gen}_{self.run_name}"
        )
        os.makedirs(save_dir, exist_ok=True)

        model_path = os.path.join(save_dir, f"gen_{gen}_id_{ind_id}.pt")

        input_shape = self.dm.input_shape
        if len(input_shape) == 3:
            input_shape = (1,) + input_shape

        device = next(LM.parameters()).device
        example_input = torch.randn(*input_shape).to(device)

        LM = LM.eval()
        traced_model = torch.jit.trace(LM.model, example_input)
        traced_model.save(model_path)

        self.logger.info(f"Saved TorchScript model to {model_path}")
        print(f"Saved model: {model_path}")




    def __getitem__(self, index):
        """
        Retrieve an individual from the population at the specified index.

        Args:
            index (int): The index of the individual to retrieve.

        Returns:
            object: The individual at the specified index in the population.
        """
        return self.population[index]


    def __len__(self):
        """
        Returns the number of individuals in the population.

        This method allows the use of the `len()` function to retrieve
        the size of the population.

        Returns:
            int: The number of individuals in the population.
        """
        return len(self.population)