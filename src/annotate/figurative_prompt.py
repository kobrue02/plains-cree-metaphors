"""
Shared figurative-language annotation prompt — the dictionary-grounded,
structured decision procedure developed and iterated on in
scripts/annotate/deepseek_label_pool.py (v1-v4), then refined by two
targeted patches validated in the grounding ablation
(scripts/annotate/ablation_llm_grounding.py; paper Section 6.1): (1)
itwewina sometimes glosses a word's sense using an English idiom itself
(e.g. maka = "speak of the devil"), so "matches a listed sense" doesn't
reliably mean "literal"; (2) many idioms/metaphors are constructional
(personification, whole-clause metaphor) — no single word's sense is
"wrong," so the original word-by-word procedure never got a chance to catch
them. Provider-specific callers (src/annotate/deepseek.py, src/annotate/llm.py)
both build the user turn via src/scrapers/itwewina.py's format_for_prompt()
and send this as the system prompt, so every model is judged by the
identical procedure — no per-provider prompt drift.
"""

from __future__ import annotations
import hashlib

LABELS = ["literal", "idiom", "metaphor", "simile"]

_BASE_SYSTEM_PROMPT = """\
You are annotating Plains Cree sentences for figurative language. You are \
given the sentence's English gloss (its CONTEXTUAL MEANING — a \
meaning-for-meaning translation, not word-for-word, so a gloss that reads \
literally in English does NOT by itself mean the Cree original is literal) \
and itwêwina dictionary entries for its content words (their BASIC \
MEANINGS — the senses available for each word independent of this sentence).

First check for simile: an explicit comparison, typically marked with \
tâpiskôc ("like"/"as if"). If present, the label is simile and you can skip \
the rest of this procedure.

Otherwise, for each content word, work through this procedure. Be terse for \
words whose contextual sense obviously matches their basic sense — one short \
clause is enough ("X: matches basic sense, skip"). Save your reasoning budget \
for words that actually need the full check, since the sentence may have many \
content words:
1. Listed senses: look at ALL the senses given in the word's dictionary \
entry, not just the most basic one — dictionary entries often list more than \
one.
2. Distinctness: does the word's contribution to the sentence's CONTEXTUAL \
MEANING match ANY of its listed senses (or a more specific instance of one \
of them)? If it matches any listed sense, that word is literal — an \
already-established, lexicalized meaning is NOT a live figurative use, even \
if it isn't the sense you'd think of first. Only treat the word as distinct \
if the contextual meaning matches NONE of its listed senses.
3. If genuinely distinct from every listed sense, decide why:
   - SIMILARITY — the contextual sense and the word's basic sense belong to \
CLEARLY DIFFERENT CONCEPTUAL DOMAINS (e.g. BODY vs. LANDSCAPE, CONCRETE \
OBJECT vs. EMOTION, MOTION vs. TIME), and the contextual sense is only \
understandable as that basic sense mapped analogically onto the new domain \
— this is a METAPHOR. A shared feature is not enough by itself: if both \
senses sit in the same general domain (e.g. both are ordinary physical \
actions, or both are types of speech), that is NOT metaphor — treat it as \
literal instead, since it's just an ordinary extended/loose use, not a \
cross-domain figurative one.
   - CONVENTION — the distinctness isn't explained by a per-word domain \
mapping at all; instead the sentence contains a fixed multi-word expression \
whose overall meaning is simply the established, lexicalized sense of that \
whole expression, not built compositionally from the parts — this is an \
IDIOM.

If every content word's contextual sense matches one of its listed senses \
(step 2 never triggers), the label is literal.

You may only conclude metaphor or idiom if you can point to one specific \
word or expression and state its figurative meaning concretely in a single \
clause. If you cannot do this — if the mismatch is vague, diffuse across the \
whole sentence, or you can't settle on a specific culprit — the label must \
be literal instead.

Apply this procedure internally, then respond in EXACTLY this three-line \
format and nothing else:
LABEL: literal / metaphor / idiom / simile
EXPRESSION: the specific word or expression that is figurative (or "none" if literal)
MEANING: a one-clause paraphrase of its figurative meaning (or "none" if literal)\
"""

# Two targeted patches validated in the grounding ablation (paper Section 6.1) —
# surgical additions, not a wholesale loosening, so literal precision isn't
# sacrificed to fix idiom/metaphor recall. Applied via marker-replace (not
# hand-retyped) so this can't silently diverge from what was actually tested.
_IDIOM_GLOSS_MARKER = "listed senses."
_IDIOM_GLOSS_PATCH = (
    " Exception: if a listed \"sense\" is itself phrased as an English idiom or "
    "figurative expression (e.g. a gloss like \"speak of the devil\" rather than "
    "a plain descriptive definition), do not treat matching it as establishing "
    "literalness — judge instead whether the CREE word's usage here is itself "
    "conventionalized/idiomatic, independent of how the dictionary happens to "
    "phrase its English gloss."
)
_CONSTRUCTIONAL_MARKER = "Otherwise, for each content word, work through this procedure."
_CONSTRUCTIONAL_PATCH = (
    "Also check, independent of any single word: does the sentence as a whole "
    "depict something impossible or unusual taken literally — e.g. an inanimate "
    "object or non-human entity acting with human intention, emotion, or agency "
    "(personification), or an event framed as impossible under ordinary "
    "circumstances? If so, this can indicate metaphor or idiom even when every "
    "individual word matches its listed sense; identify the specific predication "
    "responsible.\n\n"
    + _CONSTRUCTIONAL_MARKER
)

assert _BASE_SYSTEM_PROMPT.count(_IDIOM_GLOSS_MARKER) == 1, \
    "idiom-gloss patch anchor not found exactly once — prompt wording changed upstream?"
assert _BASE_SYSTEM_PROMPT.count(_CONSTRUCTIONAL_MARKER) == 1, \
    "constructional patch anchor not found exactly once — prompt wording changed upstream?"

SYSTEM_PROMPT = (
    _BASE_SYSTEM_PROMPT
    .replace(_IDIOM_GLOSS_MARKER, _IDIOM_GLOSS_MARKER + _IDIOM_GLOSS_PATCH, 1)
    .replace(_CONSTRUCTIONAL_MARKER, _CONSTRUCTIONAL_PATCH, 1)
)
assert _IDIOM_GLOSS_PATCH in SYSTEM_PROMPT, "idiom-gloss patch failed to apply"
assert _CONSTRUCTIONAL_PATCH in SYSTEM_PROMPT, "constructional patch failed to apply"


def prompt_version(prompt: str) -> str:
    """Short fingerprint of a system prompt string. Stamped onto every cached
    annotation (see annotate_pool() / ablation_llm_grounding.py's load_cache())
    so a prompt change (e.g. the idiom-gloss/constructional patches above)
    invalidates stale cache entries automatically instead of silently serving
    pre-patch labels forever — this bit us once (deepseek_on_gold_* files from
    before the patch existed kept looking current after rerunning)."""
    return hashlib.sha256(prompt.encode()).hexdigest()[:10]


def parse_label(content: str) -> str | None:
    """Extract the LABEL: line from a structured response. None if missing/invalid."""
    for line in (content or "").splitlines():
        line = line.strip()
        if line.lower().startswith("label:"):
            candidate = line.split(":", 1)[1].strip().lower()
            return candidate if candidate in LABELS else None
    return None
