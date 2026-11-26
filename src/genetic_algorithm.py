
"""Genetic algorithm for SVM feature selection and hyperparameter search."""

import random

import numpy as np

from sklearn.base import clone
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, GroupKFold

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class GeneticSVMOptimizer:

    C_VALUES = [0.1, 1.0, 10.0, 100.0]
    GAMMA_VALUES = ["scale", 0.001, 0.01, 0.1]

    def __init__(
        self,
        population_size=8,
        generations=4,
        mutation_rate=0.08,
        cv_folds=2,
        random_state=42
    ):
        if population_size < 4:
            raise ValueError("Population size must be at least 4.")

        if generations < 1:
            raise ValueError("At least one generation is required.")

        if not 0 <= mutation_rate <= 1:
            raise ValueError("Mutation rate must be between 0 and 1.")

        if cv_folds < 2:
            raise ValueError("At least two CV folds are required.")

        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.cv_folds = cv_folds
        self.random_state = random_state

        self.rng = random.Random(random_state)

        self.best_solution = None
        self.best_fitness = -1.0
        self.history = []

    def _create_chromosome(self, n_features):
        mask = [
            self.rng.random() < 0.5
            for _ in range(n_features)
        ]

        if not any(mask):
            mask[self.rng.randrange(n_features)] = True

        return {
            "mask": mask,
            "c_index": self.rng.randrange(len(self.C_VALUES)),
            "gamma_index": self.rng.randrange(len(self.GAMMA_VALUES))
        }

    def _repair(self, chromosome):
        if not any(chromosome["mask"]):
            chromosome["mask"][
                self.rng.randrange(len(chromosome["mask"]))
            ] = True

        return chromosome

    def _fitness(self, chromosome, X, y, splits):
        mask = np.asarray(chromosome["mask"], dtype=bool)

        selected_features = X[:, mask]

        c_value = self.C_VALUES[chromosome["c_index"]]
        gamma_value = self.GAMMA_VALUES[
            chromosome["gamma_index"]
        ]

        scores = []

        for train_indices, validation_indices in splits:
            model = make_pipeline(
                StandardScaler(),
                SVC(
                    kernel="rbf",
                    C=c_value,
                    gamma=gamma_value,
                    class_weight="balanced"
                )
            )

            model.fit(
                selected_features[train_indices],
                y[train_indices]
            )

            predictions = model.predict(
                selected_features[validation_indices]
            )

            scores.append(
                f1_score(
                    y[validation_indices],
                    predictions,
                    average="macro",
                    zero_division=0
                )
            )

        average_score = float(np.mean(scores))

        feature_ratio = np.mean(mask)

        penalty = 0.02 * feature_ratio

        return average_score - penalty

    def _select_parent(self, population, fitnesses):
        candidates = self.rng.sample(
            range(len(population)),
            k=min(3, len(population))
        )

        best_index = max(
            candidates,
            key=lambda index: fitnesses[index]
        )

        return population[best_index]

    def _crossover(self, parent1, parent2):
        child1 = {
            "mask": [],
            "c_index": parent1["c_index"],
            "gamma_index": parent1["gamma_index"]
        }

        child2 = {
            "mask": [],
            "c_index": parent2["c_index"],
            "gamma_index": parent2["gamma_index"]
        }

        for gene1, gene2 in zip(
            parent1["mask"],
            parent2["mask"]
        ):
            if self.rng.random() < 0.5:
                child1["mask"].append(gene1)
                child2["mask"].append(gene2)
            else:
                child1["mask"].append(gene2)
                child2["mask"].append(gene1)

        if self.rng.random() < 0.5:
            child1["c_index"], child2["c_index"] = (
                child2["c_index"],
                child1["c_index"]
            )

        if self.rng.random() < 0.5:
            child1["gamma_index"], child2["gamma_index"] = (
                child2["gamma_index"],
                child1["gamma_index"]
            )

        return child1, child2

    def _mutate(self, chromosome):
        for index in range(len(chromosome["mask"])):
            if self.rng.random() < self.mutation_rate:
                chromosome["mask"][index] = (
                    not chromosome["mask"][index]
                )

        if self.rng.random() < self.mutation_rate:
            chromosome["c_index"] = self.rng.randrange(
                len(self.C_VALUES)
            )

        if self.rng.random() < self.mutation_rate:
            chromosome["gamma_index"] = self.rng.randrange(
                len(self.GAMMA_VALUES)
            )

        return self._repair(chromosome)

   
    def fit(self, X, y, groups=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim != 2 or len(X) != len(y):
            raise ValueError("Invalid feature or label dimensions.")

        if len(X) == 0 or X.shape[1] == 0:
            raise ValueError("Dataset must contain samples and features.")

        if not np.isfinite(X).all():
            raise ValueError("Features contain NaN or infinite values.")

        if len(np.unique(y)) < 2:
            raise ValueError("At least two classes are required.")

        if groups is not None:
            groups = np.asarray(groups)

            if len(groups) != len(y):
                raise ValueError("Groups and labels must have equal lengths.")

            if len(np.unique(groups)) < self.cv_folds:
                raise ValueError("Not enough groups for cross-validation.")

            cv = GroupKFold(n_splits=self.cv_folds)
            splits = list(cv.split(X, y, groups))

        else:
            if min(np.unique(y, return_counts=True)[1]) < self.cv_folds:
                raise ValueError(
                    "Each class must have at least cv_folds samples."
                )

            cv = StratifiedKFold(
                n_splits=self.cv_folds,
                shuffle=True,
                random_state=self.random_state
            )

            splits = list(cv.split(X, y))

        population = [
            self._create_chromosome(X.shape[1])
            for _ in range(self.population_size)
        ]

        self.history = []
        self.best_solution = None
        self.best_fitness = -1.0

        for generation in range(self.generations):
            fitnesses = [
                self._fitness(chromosome, X, y, splits)
                for chromosome in population
            ]

            generation_best = int(np.argmax(fitnesses))
            score = fitnesses[generation_best]

            if score > self.best_fitness:
                self.best_fitness = score
                self.best_solution = {
                    "mask": population[generation_best]["mask"][:],
                    "c_index": population[generation_best]["c_index"],
                    "gamma_index": population[generation_best]["gamma_index"]
                }

            self.history.append({
                "generation": generation + 1,
                "best_fitness": float(score)
            })

            ranked = sorted(
                zip(population, fitnesses),
                key=lambda item: item[1],
                reverse=True
            )

            next_population = [
                {
                    "mask": item[0]["mask"][:],
                    "c_index": item[0]["c_index"],
                    "gamma_index": item[0]["gamma_index"]
                }
                for item in ranked[:2]
            ]

            while len(next_population) < self.population_size:
                parent1 = self._select_parent(population, fitnesses)
                parent2 = self._select_parent(population, fitnesses)

                child1, child2 = self._crossover(parent1, parent2)

                next_population.append(self._mutate(child1))

                if len(next_population) < self.population_size:
                    next_population.append(self._mutate(child2))

            population = next_population

        return self


    def get_best_parameters(self):
        if self.best_solution is None:
            raise RuntimeError("Call fit() before requesting parameters.")

        mask = np.asarray(
            self.best_solution["mask"],
            dtype=bool
        )

        return {
            "selected_features": np.flatnonzero(mask).tolist(),
            "number_of_features": int(mask.sum()),
            "total_features": len(mask),
            "C": self.C_VALUES[self.best_solution["c_index"]],
            "gamma": self.GAMMA_VALUES[
                self.best_solution["gamma_index"]
            ],
            "fitness": self.best_fitness,
            "history": list(self.history)
        }

    def build_model(self):
        if self.best_solution is None:
            raise RuntimeError("Run fit() first.")

        return make_pipeline(
            StandardScaler(),
            SVC(
                kernel="rbf",
                C=self.C_VALUES[self.best_solution["c_index"]],
                gamma=self.GAMMA_VALUES[
                    self.best_solution["gamma_index"]
                ],
                class_weight="balanced"
            )
        )
