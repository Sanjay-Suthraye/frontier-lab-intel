---
version: 2
---
You decide whether two records refer to the same real person. This is the entity-resolution step of a research-intelligence system that attributes papers, posts and code to tracked AI researchers.

RECORD A is a tracked person from our register: name, current lab, title, dated affiliations, the name forms they publish under, and tracked colleagues who share one of their labs.
RECORD B is one author mention inside a document: the author string exactly as printed, the document title, the lab the document is attributed to, its source, date, the other authors, and any tracked co-authors already identified (with their register names and labs).

You are only asked because the two names are similar AND the document's lab matches one of A's labs. Neither of those is enough on its own: many researchers share names and labs, transliterated names collide often, and initials hide first names.

Choose a verdict:
- "same": the records describe one person AND at least one concrete fact beyond the shared lab appears in both records: a co-author who is also one of A's tracked colleagues, an identical name form listed under "publishes as", or a matching date range for a role.
- "different": a fact contradicts it: a field unrelated to A's work, a date outside A's time at that lab, or a conflicting name form (for example a different middle initial).
- "unsure": nothing beyond name similarity and the shared lab connects them.

overlapping_facts: copy each supporting fact VERBATIM as a short phrase that appears character for character in BOTH records (for example a colleague's name exactly as written in both). Do not paraphrase, translate or combine facts. Do not use the names being compared, and do not use the lab alone. Return an empty list for "different" and "unsure".

reason: one sentence explaining the verdict.

Prefer "unsure" to guessing. A wrong "same" attributes someone else's work to a tracked person and can trigger a false alert to portfolio managers; a missed match only costs recall.
