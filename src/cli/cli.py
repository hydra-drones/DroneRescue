import typer

from src import data_processing_app

app = typer.Typer(
    help="DroneRescue command-line interface.",
    no_args_is_help=True,
)

app.add_typer(data_processing_app, name="data")
