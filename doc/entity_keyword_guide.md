# Static semantic keyword guide

`data/entity_semantic_keywords.json` is the only keyword source used by the
current pipeline. It is a curated, versioned semantic lexicon; the pipeline
does not derive, rank, or learn keywords from query records at runtime.

The lexicon separates classes by the object requested:

| Entity | Object | Examples of semantic evidence |
|---|---|---|
| CDR | Communication event | call, SMS, email, call record, telecom event |
| Phone | Device/number/subscriber | handset, MSISDN, SIM, mobile device |
| EVisa Request | Travel/entry record | visa, passport, border, citizenship, entry permit |
| Person | Civil identity | individual, birth date, occupation, gender |
| Insight | Analytic finding | assessment, indicator, observation, threat finding |
| Investigation | Managed case | case status, priority, evidence, investigator |
| Report | Written document | report content, narrative, briefing, summary |
| Web Activity | Content event | post, comment, URL, sentiment, video |
| Web Actor | Online identity | profile, account, channel, follower, handle |

At inference, phrase matches do not independently decide a class. They apply a
soft boost to that class's logistic probability. A static content-author
relation rule additionally boosts both Web Activity and Web Actor when content
evidence occurs with an author phrase such as `made by` or `posted by`.
Strictly removing all un-matched classes was evaluated and rejected because the lexicon
cannot cover every natural-language formulation. See
`doc/entity_extraction_report.md` for the formula, ablation, and metrics.

## Contextual enrichment and relation evidence

The lexicon now additionally distinguishes CDR activity (`made calls`, `phone
activity`, `telecom activity`), EVisa entry (`entered the country`, `arrived
from`, `border crossing`), content sentiment (`post sentiment`, `comment
sentiment`), and actor identity (`comment author`, `commenter`, `follower
count`, `sentiment score`). Broad `sentiment` was removed from the Web Activity
base terms because it incorrectly captures actor-profile sentiment queries.

The content-author rule activates only when both conditions hold: at least one
Web Activity phrase is present and an author phrase is present. It softly raises
the probabilities of both Web Activity and Web Actor; it does not force either
class. All phrase matching uses true word boundaries, so `male` does not match
the substring inside `female`.

The additions and relation rule were rerun through validation-only
optimization. A subsequent user-directed test-error review adjusted the final
relation boost; its resulting test score is documented as post-test-tuning,
not as a blind estimate, in the report. The final configuration and latency
artifact are in `artifacts/entity-final-001/`.
