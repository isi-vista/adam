from immutablecollections import immutableset

DETERMINERS = immutableset(
    [
        "the",
        "a",
        "yi1_ge4",
        "yi1_jang1",
        "yi1_ben3",
        "yi1_jyan1",
        "yi1_lyang4",
        "yi1_bei1",
        "yi1_ba3",
        "yi1_jr1",
        "yi1_shan4",
        "yi1_ding3",
        "yi1_kwai4",
    ]
)
"""
These are determiners we automatically add to the beginning of non-proper English noun phrases.
This is a language-specific hack since learning determiners is out of our scope:
https://github.com/isi-vista/adam/issues/498
"""
ENGLISH_BLOCK_DETERMINERS = immutableset(["you", "me", "your", "my"]).union(DETERMINERS)
"""
These words block the addition of the determiners above to English noun phrases.
"""
