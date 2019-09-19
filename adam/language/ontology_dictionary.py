r"""
Mappings from `Ontology`\ s to particular languages.
"""

from attr import attrib, attrs
from attr.validators import instance_of
from immutablecollections import ImmutableSet, ImmutableSetMultiDict
from immutablecollections.converter_utils import _to_immutablesetmultidict
from vistautils.preconditions import check_arg

from adam.language.lexicon import LexiconEntry
from adam.ontology import OntologyNode
from adam.ontology.ontology import Ontology


@attrs(frozen=True, slots=True)
class OntologyLexicon:
    r"""
    A mapping from `OntologyNode`\ s to words.

    This will become more sophisticated in the future.
    """
    ontology: Ontology = attrib(validator=instance_of(Ontology))
    """
    The ontology this lexicon assigns words to.
    """
    _ontology_node_to_word: ImmutableSetMultiDict[OntologyNode, LexiconEntry] = attrib(
        converter=_to_immutablesetmultidict
    )
    r"""
    Maps `OntologyNode`\ s to `LexiconEntry`\ s which describe them in some particular language.
    """

    def words_for_node(self, node: OntologyNode) -> ImmutableSet[LexiconEntry]:
        """
        Get the translation for an `OntologyNode`, if available.

        Args:
            node: The `OntologyNode` whose translation you want.

        Returns:
            The translation, if available.
        """
        return self._ontology_node_to_word[node]

    def __attrs_post_init__(self) -> None:
        for node in self._ontology_node_to_word:
            check_arg(
                node in self.ontology,
                f"Ontology lexicon refers to non-ontology node {node}",
            )
