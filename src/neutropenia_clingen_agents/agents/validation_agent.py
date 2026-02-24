import json
import logging
import re
from collections.abc import Collection, Mapping, Sequence
from functools import partial

from more_itertools import one

from ..utils.serialization import remove_non_printable_characters
from .state_model import (
    ClingenAgentState,
    ClinGenMention,
    Document,
    DocumentSection,
    Sentence,
)

logger = logging.getLogger(__name__)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


class ValidationAgent:
    def __init__(
        self, attributes: Collection[str] = {"VAF", "SYNTAX_N", "SYNTAX_P", "TYPE"}
    ) -> None:
        self.attributes = attributes

    @staticmethod
    def parse_json(sample: str) -> Mapping[str, Sequence[str]] | None:
        try:
            return json.loads(sample)
        except Exception:
            return None

    @staticmethod
    def json_non_empty(sample_json: Mapping[str, Sequence[str]]) -> bool:
        return len(sample_json) > 0

    @staticmethod
    def select_span(subsample: str, sample: str) -> tuple[int, int] | None:
        spans = [
            re_match.span()
            for re_match in re.finditer(subsample.lower(), sample.lower())
        ]
        try:
            return one(spans, too_short=ValueError, too_long=IndexError)
        except ValueError:
            logger.warning(
                "No instances of %s found in %s, presumed hallucinatory",
                subsample,
                sample,
            )
            return None
        except IndexError:
            logger.warning(
                "Multiple instances of %s found in %s, returning first",
                subsample,
                sample,
            )
            return spans[0]

    @staticmethod
    def select_non_hallucinatory_attribute(
        attribute_name: str, sample: str, sample_json: Mapping[str, Sequence[str]]
    ) -> tuple[int, int] | None:
        attributes = sample_json.get(attribute_name, [])
        if isinstance(attributes, str):
            logger.warning(
                "Mention %s for attribute %s does not follow singleton list schema - checking anyway",
                attributes,
                attribute_name,
            )
            return ValidationAgent.select_span(attributes, sample)
        if not isinstance(attributes, list) and not all(
            isinstance(attribute, str) for attribute in attributes
        ):
            logger.warning(
                "Malformed mention %s for attribute %s", attributes, attribute_name
            )
            return None
        try:
            single_attribute = one(
                attributes, too_short=IndexError, too_long=ValueError
            )
            return ValidationAgent.select_span(single_attribute, sample)
        except IndexError:
            return None
        except ValueError:
            logger.warning(
                "Mention for attribute %s does not follow singleton list schema - has %d elements - selecting first element which is valid",
                attribute_name,
                len(attributes),
            )
            select_span = partial(ValidationAgent.select_span, sample=sample)
            return next(filter(lambda s: s is not None, map(select_span, attributes)))

    @staticmethod
    def build_attribute_validated_json(
        attributes: Collection[str],
        sample: str,
        sample_json: Mapping[str, Sequence[str]],
    ) -> Mapping[str, tuple[int, int] | None]:
        return {
            attribute_name: ValidationAgent.select_non_hallucinatory_attribute(
                attribute_name=attribute_name,
                sample=sample,
                sample_json=sample_json,
            )
            for attribute_name in attributes
        }

    @staticmethod
    def get_validated_mention_json(
        sample: str, attributes: Collection[str]
    ) -> Mapping[str, tuple[int, int] | None] | None:
        sample_json = ValidationAgent.parse_json(sample)
        if sample_json is None:
            return None
        if not ValidationAgent.json_non_empty(sample_json):
            return None
        return ValidationAgent.build_attribute_validated_json(
            attributes=attributes, sample=sample, sample_json=sample_json
        )

    @staticmethod
    def is_heterozygous(
        vaf: str | None, threshold: int = 50, pattern: str = r"([0-9]{2}(\.[0-9]+)?%)"
    ) -> bool | None:
        if vaf is None:
            return None
        if remove_non_printable_characters(vaf.strip()).lower() == "heterozygous":
            return True

        def get_text(re_match: re.Match) -> str:
            first, second = re_match.span()
            return vaf[first:second]

        match_texts = [
            get_text(re_match) for re_match in re.finditer(pattern=pattern, string=vaf)
        ]
        try:
            vaf_percentage = one(
                match_texts,
                too_short=ValueError,
                too_long=IndexError,
            )
            return float(vaf_percentage.rstrip("%")) > threshold
        except ValueError:
            logger.warning(
                "No percentage formatted strings found in VAF finding: %s", vaf
            )
            return None
        except IndexError:
            logger.warning(
                "More than one percentage formatted strings found in VAF finding, returning maximum: %s",
                vaf,
            )

            return max(map(float, match_texts)) > threshold

    @staticmethod
    def get_clingen_mention(
        sentence: str, section_header: str, attributes: Collection[str]
    ) -> ClinGenMention | None:
        validated_mention_json = ValidationAgent.get_validated_mention_json(
            sentence, attributes
        )
        if validated_mention_json is None:
            return None
        gene = validated_mention_json.get("GENE")
        if gene is not None and any(
            validated_mention_json.get(attribute) is not None
            for attribute in attributes
            if attribute != "GENE"
        ):
            vaf_offsets = validated_mention_json.get("VAF")
            vaf_str = (
                None
                if vaf_offsets is None
                else sentence[vaf_offsets[0] : vaf_offsets[1]]
            )
            type_offsets = validated_mention_json.get("TYPE")
            type_str = (
                None
                if type_offsets is None
                else sentence[type_offsets[0] : type_offsets[1]]
            )
            type_packet = (
                type_str if type_str is not None else section_header,
                type_offsets if type_offsets is not None else None,
            )
            return ClinGenMention(
                source_text=sentence,
                gene=gene,
                syntax_n=validated_mention_json.get("SYNTAX_N"),
                syntax_p=validated_mention_json.get("SYNTAX_P"),
                vaf=validated_mention_json.get("VAF"),
                variant_type=type_packet,
                heterozygous=ValidationAgent.is_heterozygous(vaf_str),
            )

    @staticmethod
    def parse_sentence(
        sentence: Sentence, section_header: str, attributes: Collection[str]
    ) -> Sentence:
        if sentence.mention is not None:
            raise ValueError("sentence is already populated with mention")
        return Sentence(
            offsets=sentence.offsets,
            sentence_string=sentence.sentence_string,
            raw_output=sentence.raw_output,
            mention=ValidationAgent.get_clingen_mention(
                sentence=sentence.sentence_string,
                section_header=section_header,
                attributes=attributes,
            ),
        )

    @staticmethod
    def parse_section(
        document_section: DocumentSection, attributes: Collection[str]
    ) -> DocumentSection:
        section_header = document_section.section_header
        offsets = document_section.offsets
        return DocumentSection(
            section_header=section_header,
            offsets=offsets,
            sentences=[
                ValidationAgent.parse_sentence(
                    sentence=sentence,
                    section_header=section_header,
                    attributes=attributes,
                )
                for sentence in document_section.sentences
            ],
        )

    @staticmethod
    def parse_document(document: Document, attributes: Collection[str]) -> Document:
        return Document(
            file_id=document.file_id,
            sections=[
                ValidationAgent.parse_section(
                    document_section=section, attributes=attributes
                )
                for section in document.sections
            ],
        )

    def __call__(self, agent_state: ClingenAgentState) -> ClingenAgentState:
        return ClingenAgentState(
            documents=[
                ValidationAgent.parse_document(
                    document=document, attributes=self.attributes
                )
                for document in agent_state.documents
            ]
        )
