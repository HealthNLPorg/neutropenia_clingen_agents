import argparse
import os
import pathlib
import pickle
import re
from itertools import takewhile
from typing import Literal, cast

import pandas as pd
from more_itertools import flatten, partition
from numpy import nan

argparser = argparse.ArgumentParser(description="")
argparser.add_argument("--scnir_dir", type=str)
argparser.add_argument("--finished_tsv", type=str)


CORPUS = Literal["SDS", "SCNIR"]

SDS_PICKLE = (
    "~/neutropenia_corpus/cTAKES_Neutropenia_and_Pts/SDS_cTAKES/sds_table.pickle"
)
SCNIR_PICKLE = (
    "~/neutropenia_corpus/cTAKES_Neutropenia_and_Pts/SCNIR_cTAKES/scnir_table.pickle"
)


def load_pickle(
    fn: str,
) -> dict[int | str, dict[str, str | float | int | pd.Timestamp]]:
    with open(fn, mode="rb") as handle:
        unserialized_dict = pickle.load(handle)
    return unserialized_dict


def get_stem(fn: str) -> str:
    return pathlib.Path(fn).stem


def get_original_filename(row: pd.Series) -> str:
    return "_".join(row.Filename.split("_")[1:])


def get_corpus_set(corpus_dir: str) -> set[str]:
    return {get_stem(fn) for fn in os.listdir(corpus_dir)}


def complete_frame(frame: pd.DataFrame, pickle_n: str) -> pd.DataFrame:
    MRN_to_PT = load_pickle(pickle_n)
    pt_columns = set(flatten(v.keys() for v in MRN_to_PT.values()))

    def get_pdf_process(row: pd.Series) -> str:
        return row.Filename.split("_")[0]

    def get_mrn(row: pd.Series) -> int | str:
        _str_mrn = "".join(takewhile(str.isnumeric, row.Filename.split("_")[1]))
        if len(_str_mrn) > 0:
            mrn = int(_str_mrn)
            if mrn == 7107993:
                return 71079936  # not terribly impressed with the flukes in this retrieval system
            return mrn
        # bad naming for the one patient in SDS corpus who doesn't have a BCH MRN,
        # there are no patients with this issue in SCNIR
        return next(mrn for mrn in MRN_to_PT.keys() if type(mrn) is str)

    frame["PDFProcess"] = frame.apply(get_pdf_process, axis=1)
    frame["MRN"] = frame.apply(get_mrn, axis=1)
    frame["Filename"] = frame.apply(get_original_filename, axis=1)

    def clean_diagnosis(diagnosis: str) -> str:
        return re.sub("\\s+", " ", diagnosis).strip()

    for pt_column in pt_columns:

        def get_column_attr(row: pd.Series) -> str | float | int | pd.Timestamp:
            PT_info = MRN_to_PT.get(row.MRN)
            if PT_info is None:
                numeric_mrns, string_mrns = partition(
                    lambda s: type(s) is int, MRN_to_PT.keys()
                )
                mrns = [*sorted(string_mrns), *sorted(numeric_mrns)]
                raise Exception(
                    f"Missing Table Entry for mrn {row.MRN} from filename {row.Filename} in:\n\n{mrns}"
                )
            _column_info = cast(str, PT_info.get(pt_column, nan))
            if pt_column == "Diagnosis":
                return clean_diagnosis(_column_info)
            return _column_info

        frame[pt_column] = frame.apply(get_column_attr, axis=1)
    return frame


def coordinate_other_column(sub_frame: pd.DataFrame, corpus: CORPUS) -> pd.DataFrame:
    match corpus:
        case "SCNIR":
            return sds_coordinate_scnir_columns(sub_frame)
        case "SDS":
            return scnir_coordinate_sds_columns(sub_frame)


def sds_coordinate_scnir_columns(scnir_frame: pd.DataFrame) -> pd.DataFrame:
    scnir_frame["DF MRN (1)"] = scnir_frame["DF MRN"]
    scnir_frame["DF MRN (2)"] = nan
    scnir_frame.drop(columns=["DF MRN", "CORPUS"])
    return scnir_frame


# keeping the method just in case
def scnir_coordinate_sds_columns(sds_frame: pd.DataFrame) -> pd.DataFrame:
    sds_frame.drop(columns=["CORPUS"])
    return sds_frame


def process(
    scnir_pickle: str,
    sds_pickle: str,
    scnir_dir: str,
    sds_dir: str,
    finished_tsv: str,
) -> None:
    original_stem = get_stem(finished_tsv)
    finished_frame = pd.read_csv(finished_tsv, sep="\t")
    corpus_to_pickle = {"SDS": sds_pickle, "SCNIR": scnir_pickle}
    # save lookups since fewer entries here
    scnir_corpus = get_corpus_set(scnir_dir)

    def filename_to_corpus(row: pd.Series) -> CORPUS:
        if get_original_filename(row) in scnir_corpus:
            return "SCNIR"
        return "SDS"

    finished_frame["Corpus"] = finished_frame.apply(filename_to_corpus, axis=1)
    corpus_to_frame = {
        corpus: frame
        for corpus, frame in finished_frame.groupby(finished_frame["Corpus"])
    }

    def get_processed_frame(corpus: CORPUS) -> pd.DataFrame:
        core_corpus_frame = complete_frame(
            corpus_to_frame[corpus], corpus_to_pickle[corpus]
        )
        return coordinate_other_column(core_corpus_frame, corpus)

    full_frame = pd.concat(
        (get_processed_frame(corpus) for corpus in corpus_to_frame.keys()),
        ignore_index=True,
    )
    full_frame.to_csv(f"./pt_info_{original_stem}.tsv", sep="\t")
    full_frame.to_excel(f"./pt_info_{original_stem}.xlsx")


def main():
    args = argparser.parse_args()
    process(
        SCNIR_PICKLE,  # args.scnir_pickle,
        SDS_PICKLE,  # args.sds_pickle,
        args.scnir_dir,
        args.finished_tsv,
    )


if __name__ == "__main__":
    main()
