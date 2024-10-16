import glob
import os

import pandas as pd

ellipsis = "..."


def get_instance(gene: str, window: str) -> str:
    # not resistant to duplicates but just for now
    partitions = window.split(gene)
    return (
        ellipsis
        + partitions[0]
        + f"<e> {gene} </e>"
        + "".join(partitions[1:])
        + ellipsis
    )


def process_table(filename: str) -> None:
    df = pd.read_csv(filename, sep="|")
    processed = pd.Series(
        get_instance(gene, window)
        for gene, window in zip(df[" Annotation Text "], df[" Window "])
        if isinstance(gene, str) and isinstance(window, str)
    )
    fn_base = os.path.basename(filename)
    example_id = "_".join(fn_base.split("_")[:2])
    processed.to_csv(f"./{example_id}_processed.BSV", sep="|", index=None, header=False)


def driver() -> None:
    cwd = os.getcwd()
    for fn_base in glob.glob("*[!span]_table.BSV"):
        process_table(os.path.join(cwd, fn_base))


def main() -> None:
    driver()


if __name__ == "__main__":
    main()
