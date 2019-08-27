r"""
Mappings from `Ontology`\ s to particular languages.
"""

from attr import attrib, attrs
from immutablecollections import ImmutableSet, ImmutableSetMultiDict
from immutablecollections.converter_utils import _to_immutablesetmultidict

from adam.language.lexicon import LexiconEntry
from adam.ontology import OntologyNode


@attrs(frozen=True, slots=True)
class OntologyLexicon:
    r"""
    A mapping from `OntologyNode`\ s to words.

    This will become more sophisticated in the future.
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
