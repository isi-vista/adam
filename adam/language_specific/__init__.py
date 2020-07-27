"""Define language independent properties at the module level"""
from adam.language.dependency import MorphosyntacticProperty
from adam.language.lexicon import LexiconProperty

# Define universal morphosyntactic properties
FIRST_PERSON = MorphosyntacticProperty("1p")
SECOND_PERSON = MorphosyntacticProperty("2p")
THIRD_PERSON = MorphosyntacticProperty("3p")
NOMINATIVE = MorphosyntacticProperty("nom")
ACCUSATIVE = MorphosyntacticProperty("acc")

# Define universal lexicon properties
MASS_NOUN = LexiconProperty("mass-noun")
ALLOWS_DITRANSITIVE = LexiconProperty("allows-ditransitive")
