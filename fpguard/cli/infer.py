import typer

app = typer.Typer()


@app.command()
def main():
    typer.echo("Run inference & attribution - placeholder (to be implemented)")


if __name__ == "__main__":
    app()


