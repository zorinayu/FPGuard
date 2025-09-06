import typer

app = typer.Typer()


@app.command()
def main():
    typer.echo("Train projection heads - placeholder (to be implemented)")


if __name__ == "__main__":
    app()


