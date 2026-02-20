import json
import logging
from collections.abc import Collection, Mapping, Sequence
from functools import partial

from more_itertools import first_true, one

from ..utils.serialization import aggresive_normalize

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
