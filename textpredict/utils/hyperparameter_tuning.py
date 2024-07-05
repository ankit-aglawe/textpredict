# textpredict/utils/hyperparameter_tuning.py

import logging

import optuna

logger = logging.getLogger(__name__)


def objective(trial, model_class, train_data, eval_data, **kwargs):
    """
    Objective function for hyperparameter tuning.

    Args:
        trial (optuna.trial.Trial): A single trial in the hyperparameter optimization process.
        model_class (type): The model class to be tuned.
        train_data (Dataset): The training data.
        eval_data (Dataset): The evaluation data.

    Returns:
        float: The evaluation metric to be minimized.
    """
    try:
        # Example hyperparameters to tune
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)
        batch_size = trial.suggest_int("batch_size", 16, 64)

        # Initialize model with trial parameters
        model = model_class(
            learning_rate=learning_rate, batch_size=batch_size, **kwargs
        )
        model.train(train_data)
        eval_metrics = model.evaluate(eval_data)

        # The objective is to minimize the evaluation metric (e.g., loss or error)
        return eval_metrics["loss"]
    except Exception as e:
        logger.error(f"Error in objective function: {e}")
        raise


def tune_hyperparameters(model_class, train_data, eval_data, n_trials=50, **kwargs):
    """
    Tune hyperparameters using Optuna.

    Args:
        model_class (type): The model class to be tuned.
        train_data (Dataset): The training data.
        eval_data (Dataset): The evaluation data.
        n_trials (int, optional): The number of trials for the optimization. Defaults to 50.

    Returns:
        dict: The best set of hyperparameters found.
    """
    try:
        study = optuna.create_study(direction="minimize")
        study.optimize(
            lambda trial: objective(
                trial, model_class, train_data, eval_data, **kwargs
            ),
            n_trials=n_trials,
        )

        logger.info(f"Best hyperparameters: {study.best_params}")
        return study.best_params
    except Exception as e:
        logger.error(f"Error during hyperparameter tuning: {e}")
        raise
