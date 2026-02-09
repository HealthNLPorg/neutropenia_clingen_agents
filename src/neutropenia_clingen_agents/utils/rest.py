from collections.abc import Mapping

from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def read_root(message: str):
    return lambda: {"message": message}
