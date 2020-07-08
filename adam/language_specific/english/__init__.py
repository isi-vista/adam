from immutablecollections import immutableset

ENGLISH_DETERMINERS = immutableset(["the", "a"])
"""
These are determiners we automatically add to the beginning of non-proper English noun phrases.
This is a language-specific hack since learning determiners is out of our scope:
https://github.com/isi-vista/adam/issues/498
"""
ENGLISH_BLOCK_DETERMINERS = immutableset(["you", "me", "your", "my"]).union(
    ENGLISH_DETERMINERS
)
"""
These words block the addition of the determiners above to English noun phrases.
"""
