import typer

from src import download_folder_to_temp
from src import create_db, connect_to_db
from src import load_dataset_to_db
from src import process_dataset_cli

app = typer.Typer(
    help="DroneRescue command-line interface.",
    no_args_is_help=True,
)

data_app = typer.Typer(help="Data-related operations.")
data_app.command("load-from-grive")(download_folder_to_temp)
data_app.command("create-db")(create_db)
data_app.command("connect-to-db")(connect_to_db)
data_app.command("load-samples-into-db")(load_dataset_to_db)
data_app.command("process-dataset")(process_dataset_cli)


app.add_typer(data_app, name="data")
