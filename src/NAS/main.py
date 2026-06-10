import configparser
import pandas as pd
pd.set_option('display.max_colwidth', None)

import torch
import pytorch_lightning as pl
import numpy as np
from src.NAS.core.population import Population
from src.data_loader.data_loader import PhiSatSegDataModule, Sentinel2SegDataModule

import argparse, os, sys
cwd = os.getcwd()

bs = 32
nw = 4
# Argument parser
parser = argparse.ArgumentParser(description="Run the PyNAS genetic algorithm for neural architecture search.")
parser.add_argument('--gen', type=int, default=None, help='Generation to load and start NAS.')
parser.add_argument('--config', type=str, default='config.ini', help='Path to the configuration file.')
parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility.')
parser.add_argument('--max_layers', type=int, default=None, help='Maximum number of layers in the architecture.')
parser.add_argument('--max_parameters', type=int, default=None, help='Maximum number of parameters allowed in models.')
parser.add_argument('--max_iterations', type=int, default=None, help='Maximum number of generations.')
parser.add_argument('--population_size', type=int, default=None, help='Size of the population.')
parser.add_argument('--mating_pool_cutoff', type=float, default=None, help='Fraction of population to use for mating.')
parser.add_argument('--mutation_probability', type=float, default=None, help='Probability of mutation.')
parser.add_argument('--epochs', type=int, default=None, help='Number of epochs for training each model.')
parser.add_argument('--batch_size', type=int, default=None, help='Batch size for training.')
parser.add_argument('--n_random', type=int, default=None, help='Number of random individuals per generation.')
parser.add_argument('--k_best', type=int, default=None, help='Number of best individuals to keep.')
parser.add_argument('--task', type=str, default=None, help='Task type.')
parser.add_argument('--perturbation', type=str, default=None, help='Perturbation type.')
parser.add_argument('--strength', type=float, default=None, help='Strength of perturbation.')
parser.add_argument('--run_name', type=str, default=None, help='Name to classify the type of run.')
parser.add_argument(
    '--dataset',
    type=str,
    choices=['phisat2', 'sentinel2'],
    default='sentinel2',
    help='Dataset/datamodule to use: phisat2 or sentinel2.'
)

def main(args):
    try:
        config = configparser.ConfigParser()
        config.read(args.config)

        if args.seed is not None:
            config.set('Computation', 'seed', str(args.seed))
        if args.max_layers is not None:
            config.set('NAS', 'max_layers', str(args.max_layers))
        if args.max_parameters is not None:
            config.set('GA', 'max_parameters', str(args.max_parameters))
        if args.max_iterations is not None:
            config.set('GA', 'max_iterations', str(args.max_iterations))
        if args.population_size is not None:
            config.set('GA', 'population_size', str(args.population_size))
        if args.mating_pool_cutoff is not None:
            config.set('GA', 'mating_pool_cutoff', str(args.mating_pool_cutoff))
        if args.mutation_probability is not None:
            config.set('GA', 'mutation_probability', str(args.mutation_probability))
        if args.epochs is not None:
            config.set('GA', 'epochs', str(args.epochs))
        if args.batch_size is not None:
            config.set('GA', 'batch_size', str(args.batch_size))
        if args.n_random is not None:
            config.set('GA', 'n_random', str(args.n_random))
        if args.k_best is not None:
            config.set('GA', 'k_best', str(args.k_best))
        if args.task is not None:
            config.set('GA', 'task', args.task)
        if args.perturbation is not None:
            config.set('Perturbation', 'perturbation', args.perturbation)

        if args.strength is not None:
            config.set('Perturbation', 'strength', str(args.strength))

        if args.dataset == "phisat2":
            image_paths = "/shared/home/ivanderspoel/scratch/segmentation_dataset_v1/images_phisat2_npy"
            mask_paths = "/shared/home/ivanderspoel/scratch/segmentation_dataset_v1/masks_phisat2_npy"
            DataModuleClass = PhiSatSegDataModule

        elif args.dataset == "sentinel2":
            image_paths = "/shared/home/ivanderspoel/scratch/segmentation_dataset_v1/images_s2_npy"
            mask_paths = "/shared/home/ivanderspoel/scratch/segmentation_dataset_v1/masks_s2_npy"
            DataModuleClass = Sentinel2SegDataModule

        seed = config.getint('Computation', 'seed')
        pl.seed_everything(seed=seed, workers=True)
        torch.set_float32_matmul_precision("medium")

        save_dir = os.path.join(cwd, 'results')
        os.makedirs(save_dir, exist_ok=True)

        max_layers = int(config['NAS']['max_layers'])
        max_gen = int(config['GA']['max_iterations'])
        n_individuals = int(config['GA']['population_size'])
        mating_pool_cutoff = float(config['GA']['mating_pool_cutoff'])
        mutation_probability = float(config['GA']['mutation_probability'])
        epochs = int(config['GA']['epochs'])
        batch_size = int(config['GA']['batch_size'])
        n_random = int(config['GA']['n_random'])
        k_best = int(config['GA']['k_best'])
        task = str(config['GA']['task'])
        max_params = int(config['GA']['max_parameters'])
        perturbation = str(config['Perturbation']['perturbation'])
        strength = float(config['Perturbation']['strength'])
        
        if args.run_name is not None:
            run_name = args.run_name
        else:
            run_name = str(np.uint64(np.random.randint(0, 2**64, dtype=np.uint64)))
        
        
        dm = DataModuleClass(
            image_dir=image_paths,
            mask_dir=mask_paths,
            batch_size=bs,
            val_split=0.2,
            num_workers=nw,
            perturbation=perturbation,
            strength=strength
        )

        dm.setup(stage='fit')

        pop = Population(
            n_individuals=n_individuals,
            max_layers=max_layers,
            dm=dm,
            save_directory=save_dir,
            max_parameters=max_params,
            run_name=run_name,
            perturbation=perturbation,
            strength=strength
        )

        pop._use_group_norm = False

        config_path = os.path.join(pop.save_directory, 'config.ini')
        if not os.path.exists(config_path):
            with open(config_path, 'w') as configfile:
                config.write(configfile)

        if args.gen is not None:
            pop.load_generation(args.gen)
        else:
            pop.initial_poll()

        for _ in range(max_gen):
            pop.train_generation(task=task, lr=0.001, epochs=epochs, batch_size=bs)
            pop.evolve(
                mating_pool_cutoff=mating_pool_cutoff,
                mutation_probability=mutation_probability,
                k_best=k_best,
                n_random=n_random
            )

        return 0

    except Exception as e:
        print(f"An error occurred: {e}")
        return 1


if __name__ == '__main__':
    args = parser.parse_args()
    r = main(args=args)
    if r == 0:
        print("Execution completed successfully.")
    else:
        print("Execution failed.")
    sys.exit(r)