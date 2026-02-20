import json
import logging
from collections.abc import Collection, Mapping, Sequence
from functools import partial

from more_itertools import first_true, one

from ..utils.serialization import aggresive_normalize
from .state_model import ClinGenMention, ClingenAgentState

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
    def not_hallucination(subsample: str, sample: str) -> bool:
        return aggresive_normalize(subsample) in aggresive_normalize(sample)

    @staticmethod
    def select_non_hallucinatory_attribute(
        attribute_name: str, sample: str, sample_json: Mapping[str, Sequence[str]]
    ) -> str | None:
        attributes = sample_json.get(attribute_name, [])
        if isinstance(attributes, str):
            logger.warning(
                "Mention %s for attribute %s does not follow singleton list schema - checking anyway",
                attributes,
                attribute_name,
            )
            return (
                attributes
                if ValidationAgent.not_hallucination(attributes, sample)
                else None
            )
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
            return (
                single_attribute
                if ValidationAgent.not_hallucination(single_attribute, sample)
                else None
            )
        except IndexError:
            return None
        except ValueError:
            logger.warning(
                "Mention for attribute %s does not follow singleton list schema - has %d elements - selecting first element which is valid",
                attribute_name,
                len(attributes),
            )
            return first_true(
                attributes,
                pred=partial(ValidationAgent.not_hallucination, sample=sample),
            )

    @staticmethod
    def build_attribute_validated_json(
        attributes: Collection[str],
        sample: str,
        sample_json: Mapping[str, Sequence[str]],
    ) -> Mapping[str, str | None]:
        return {
            attribute_name: ValidationAgent.select_non_hallucinatory_attribute(
                attribute_name=attribute_name,
                sample=sample,
                sample_json=sample_json,
            )
            for attribute_name in attributes
        }

    def get_validated_mention_json(
        self,
        sample: str,
    ) -> Mapping[str, str | None] | None:
        sample_json = ValidationAgent.parse_json(sample)
        if sample_json is None:
            return None
        if not ValidationAgent.json_non_empty(sample_json):
            return None
        return ValidationAgent.build_attribute_validated_json(
            attributes=self.attributes, sample=sample, sample_json=sample_json
        )

    @staticmethod
    def is_heterozygous(vaf: str | None, threshold: int = 50) -> bool | None:
        # TODO fill in the heuristic
        return None

    def get_clingen_mention(
        self, sentence: str, section_header: str
    ) -> ClinGenMention | None:
        validated_mention_json = self.get_validated_mention_json(sentence)
        if validated_mention_json is None:
            return None
        if validated_mention_json.get("GENE") is not None and any(
            validated_mention_json.get(attribute) is not None
            for attribute in self.attributes
            if attribute != "GENE"
        ):
            return ClinGenMention(
                source_text=sentence,
                gene=validated_mention_json.get("GENE"),
                syntax_n=validated_mention_json.get("SYNTAX_N"),
                syntax_p=validated_mention_json.get("SYNTAX_P"),
                vaf=validated_mention_json.get("VAF"),
                variant_type=validated_mention_json.get("TYPE"),
                heterozygous=ValidationAgent.is_heterozygous(
                    validated_mention_json.get("VAF")
                ),
            )

    def __call__(self, agent_state: ClingenAgentState) -> ClingenAgentState:
        # TODO finish
        return agent_state
