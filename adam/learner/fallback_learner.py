from typing_extensions import Protocol, runtime

from adam.semantics import SyntaxSemanticsVariable, ActionSemanticNode


@runtime
class ActionFallbackLearnerProtocol(Protocol):
    def ignore_slot_internal_structure_failure(
        self,
        action_semantic_node: ActionSemanticNode,
        slot_with_failure: SyntaxSemanticsVariable,
    ) -> bool:
        """
        Given an action semantic node and a slot failing because of its internal structure,
        return whether it makes sense to ignore the internal structure of that slot.
        """
