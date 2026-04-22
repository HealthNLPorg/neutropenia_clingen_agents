from typing import Final

from pydantic import BaseModel, Field

Offsets = tuple[int, int]


class ClinGenMention(BaseModel):
    source_text: Final[str] = Field(description="Source text for the mention")
    gene: Final[tuple[int, int]] = Field(
        description="The anchor of any gene mention, the mention does not exist without a gene"
    )
    syntax_n: Final[tuple[int, int] | None] = Field(
        description="Gene variant syntax - nucleotide change"
    )
    syntax_p: Final[tuple[int, int] | None] = Field(
        description="Gene variant syntax - protein change"
    )
    vaf: Final[tuple[int, int] | None] = Field(description="Variant allele frequency")
    variant_type: Final[tuple[int, int] | None] = Field(
        description="Variant type (pathogenic, benign, unknown, etc.) and whether is parsed from source sentence (true if parsed from sentence, false if from section header)"
    )
    heterozygous: Final[bool | None] = Field(
        description="Whether the variant is heterozygous (VAF > 50%)"
    )


class Sentence(BaseModel):
    offsets: Final[Offsets]
    sentence_string: Final[str]
    raw_output: Final[str | None]
    mention: Final[ClinGenMention | None]
