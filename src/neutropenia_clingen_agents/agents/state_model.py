from collections.abc import Sequence
from typing import TypedDict

from pydantic import BaseModel, Field

Offsets = tuple[int, int]


class ClinGenMention(BaseModel):
    source_text: str = Field("Source text for the mention")
    gene: tuple[int, int] = Field(
        description="The anchor of any gene mention, the mention does not exist without a gene"
    )
    syntax_n: tuple[int, int] | None = Field(
        description="Gene variant syntax - nucleotide change"
    )
    syntax_p: tuple[int, int] | None = Field(
        description="Gene variant syntax - protein change"
    )
    vaf: tuple[int, int] | None = Field(description="Variant allele frequency")
    variant_type: tuple[str, tuple[int, int] | None] | None = Field(
        description="Variant type (pathogenic, benign, unknown, etc.) and whether is parsed from source sentence (true if parsed from sentence, false if from section header)"
    )
    heterozygous: bool | None = Field(
        description="Whether the variant is heterozygous (VAF > 50%)"
    )


class Sentence(TypedDict):
    offsets: Offsets
    sentence_string: str
    raw_output: str | None
    mention: ClinGenMention | None


class DocumentSection(TypedDict):
    section_header: str
    offsets: Offsets
    sentences: Sequence[Sentence]


class Document(TypedDict):
    file_id: str
    sections: Sequence[DocumentSection]


class ClingenAgentState(TypedDict):
    documents: Sequence[Document]
