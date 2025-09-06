import typer

app = typer.Typer()


@app.command()
def main():
    typer.echo("Detoxify training corpus - placeholder (to be implemented)")


if __name__ == "__main__":
    app()


