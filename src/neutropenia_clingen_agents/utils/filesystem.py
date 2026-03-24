import pathlib


def make_directory(dirname: str) -> None:
    pathlib.Path(dirname).mkdir(parents=True, exist_ok=True)
