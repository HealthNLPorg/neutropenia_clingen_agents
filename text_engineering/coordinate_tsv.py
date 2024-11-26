import argparse
import pickle
from typing import Literal

import pandas as pd
from more_itertools import flatten
from numpy import nan

argparser = argparse.ArgumentParser(description="")
argparser.add_argument("--scnir_pickle", type=str)
argparser.add_argument("--scnir_tsv", type=str)
argparser.add_argument("--sds_pickle", type=str)
argparser.add_argument("--sds_tsv", type=str)

CORPUS = Literal["SDS", "SCNIR"]


def load_pickle(
    fn: str,
) -> dict[int | str, dict[str, str | float | int | pd.Timestamp]]:
    with open(fn, mode="rb") as handle:
        unserialized_dict = pickle.load(handle)
    return unserialized_dict


def complete_frame(frame_n: str, pickle_n: str, corpus: CORPUS) -> pd.DataFrame:
    frame = pd.read_csv(frame_n, sep="\t")
    MRN_to_PT = load_pickle(pickle_n)
    pt_columns = set(flatten(v.keys() for v in MRN_to_PT.values()))

    def get_pdf_process(row: pd.Series) -> str:
        return row.filename.split("_")[0]

    def get_mrn(row: pd.Series) -> str:
        return row.filename.split("_")[1].split["-"][0]

    def get_original_filename(row: pd.Series) -> str:
        return "_".join(row.filename.split("_")[1:])

    frame["PDF_PROCESS"] = frame.apply(get_pdf_process)
    frame["MRN"] = frame.apply(get_mrn)
    frame["filename"] = frame.apply(get_original_filename)
    frame["CORPUS"] = corpus

    for pt_column in pt_columns:

        def get_column_attr(row: pd.Series) -> str | float | int | pd.Timestamp:
            PT_info = MRN_to_PT.get(row.MRN)
            if PT_info is None:
                raise Exception(f"Missing Table Entry for {row.MRN}")
            return PT_info.get(pt_column, nan)

        frame[pt_column] = frame.apply(get_column_attr)
    return frame


def sds_coordinate_scnir_columns(scnir_frame: pd.DataFrame) -> pd.DataFrame:
    scnir_frame["DF MRN (1)"] = scnir_frame["DF MRN"]
    scnir_frame["DF MRN (2)"] = nan
    scnir_frame.drop(columns=["DF MRN"])
    return scnir_frame


# keeping the method just in case
def scnir_coordinate_sds_columns(sds_frame: pd.DataFrame) -> pd.DataFrame:
    return sds_frame


def process(
    scnir_pickle: str,
    scnir_tsv: str,
    sds_pickle: str,
    sds_tsv: str,
) -> None:
    core_scnir_frame = complete_frame(scnir_tsv, scnir_pickle, "SCNIR")
    core_sds_frame = complete_frame(sds_tsv, sds_pickle, "SDS")
    coordinated_scnir_frame = sds_coordinate_scnir_columns(core_scnir_frame)
    coordinated_sds_frame = scnir_coordinate_sds_columns(core_sds_frame)
    full_frame = pd.concat(
        (coordinated_scnir_frame, coordinated_sds_frame), ignore_index=True
    )
    full_frame.to_csv("./full_frame.tsv", sep="\t")


def main():
    args = argparser.parse_args()
    process(
        args.scnir_pickle,
        args.scnir_tsv,
        args.sds_pickle,
        args.sds_tsv,
    )


if __name__ == "__main__":
    main()
