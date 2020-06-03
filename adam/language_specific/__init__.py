from adam.language.dependency import (
    DependencyRole,
    HEAD,
    MorphosyntacticProperty,
    PartOfSpeechTag,
    RoleOrderDependencyTreeLinearizer,
)

FIRST_PERSON = MorphosyntacticProperty("1p")
SECOND_PERSON = MorphosyntacticProperty("2p")
THIRD_PERSON = MorphosyntacticProperty("3p")
NOMINATIVE = MorphosyntacticProperty("nom")
ACCUSATIVE = MorphosyntacticProperty("acc")
