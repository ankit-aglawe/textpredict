import click

from textpredict import TextPredict
from textpredict.logger import set_logging_level
from textpredict.utils.error_handling import log_and_raise


@click.group()
def cli():
    pass


@click.command()
@click.argument("text")
@click.option(
    "--task",
    default="sentiment",
    help="The task to perform: sentiment, emotion, zeroshot, etc.",
)
@click.option("--model", default=None, help="The model to use for the task.")
@click.option(
    "--class-list",
    default=None,
    help="Comma-separated list of candidate labels for zero-shot classification.",
)
@click.option(
    "--log-level",
    default="INFO",
    help="Set the logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL.",
)
def analyze(text, task, model, class_list, log_level):
    """Analyze the given text for the specified task."""
    try:
        set_logging_level(log_level)
        tp = TextPredict(model_name=model)
        class_list = class_list.split(",") if class_list else None
        result = tp.analyse(text, task, class_list)
        click.echo(result)
    except Exception as e:
        log_and_raise(RuntimeError, f"Error during CLI analysis: {e}")


cli.add_command(analyze)

if __name__ == "__main__":
    cli()
