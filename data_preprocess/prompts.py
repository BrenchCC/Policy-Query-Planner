import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

OUTPUT_SCHEMA_TEXT = json.dumps(
    {
        "queries": [
            {
                "id": "q1",
                "query": "retrieval query",
                "depends_on": []
            }
        ]
    },
    ensure_ascii = False,
    indent = 2
)

SFT_PROMPT_TEMPLATE = """You are preparing a high-quality retrieval-query-planner training record.

Task:
Rewrite the user's policy question into one compact, standalone retrieval query. The query must preserve every constraint that can change eligibility or the applicable procedure, including age, dates, duration, employment status, family relationship, residence, income, disability, and prior actions. Resolve references from the scenario. Use the provided policy title only as topic guidance; do not copy answer-only facts.

Hard rules:
1. Return exactly one JSON object and no Markdown.
2. Use exactly one query with id q1 and an empty depends_on list.
3. Do not answer the policy question.
4. Do not invent constraints or outcomes.
5. Do not include facts that appear only in the gold answer or evidence.
6. Keep the query under 45 English words.
7. Make the query searchable as a standalone string.

Output shape:
{schema}

Example input:
Policy title: Statutory Paternity Pay and Leave
Scenario: I have worked for my employer for two months and my partner is due in twenty weeks.
Question: Will I qualify for paternity leave?

Example output:
{{"queries":[{{"id":"q1","query":"Statutory paternity leave eligibility two months employment partner due in twenty weeks","depends_on":[]}}]}}

Now process this record:
Policy title: {title}
Scenario: {scenario}
Question: {question}
"""

DPO_PROMPT_TEMPLATE = """You are creating a difficult rejected response for retrieval-query-planner preference training.

The chosen plan is correct. Produce one plausible but meaningfully worse plan with the requested error type. The rejected plan must remain valid JSON and look superficially reasonable, but it must reduce retrieval quality. Do not make it nonsensical, do not answer the question, and do not change the JSON structure.

Error types:
- unresolved_reference: keep an ambiguous pronoun or elliptical reference unresolved.
- entity_omission: omit the central named entity, policy, or relationship.
- constraint_omission: remove one eligibility-changing date, amount, duration, status, or condition.
- overly_broad: replace the focused query with a generic topic query.
- wrong_context: import a plausible but incorrect entity or constraint from the history or scenario.

Hard rules:
1. Return exactly one JSON object and no Markdown.
2. Return exactly one query with id q1 and no dependencies.
3. The rejected query must differ from the chosen query.
4. Apply only the requested primary error type.
5. Never include the gold answer.

Output shape:
{schema}

Input:
User input:
{user_input}

Chosen plan:
{chosen}

Required error type: {error_type}
"""

GRPO_PROMPT_TEMPLATE = """You are preparing a reference multi-hop retrieval plan for a public-policy question.

Task:
Decompose the scenario and question into two to four ordered retrieval queries. Each query must retrieve one necessary fact or rule. Later queries may depend on answers from earlier queries. A dependent query must use placeholders such as {{{{q1.answer}}}} and list the matching dependency in depends_on.

Hard rules:
1. Return exactly one JSON object and no Markdown.
2. Use sequential ids q1, q2, q3, q4 without gaps.
3. Dependencies may only refer to earlier queries.
4. Every dependency must appear as a placeholder in the query text.
5. Do not answer the final question.
6. Do not quote or reveal the reference answer.
7. Preserve all eligibility-changing constraints from the scenario.
8. Avoid redundant queries; each hop must retrieve distinct evidence.
9. Make every query independently executable after placeholders are substituted.

Output shape example:
{{"queries":[{{"id":"q1","query":"probate administrator priority for an intestate estate","depends_on":[]}},{{"id":"q2","query":"priority between {{{{q1.answer}}}} and a sibling of the deceased","depends_on":["q1"]}}]}}

Input:
Policy title: {title}
Scenario: {scenario}
Question: {question}

Reference evidence for teacher use only:
{evidence}
"""


def build_prompt(stage: str, payload: dict[str, Any]) -> str:
    """Render a stage-specific generation prompt.

    Args:
        stage: Generation stage name.
        payload: Source fields required by the selected template.

    Returns:
        Rendered generation prompt.

    Raises:
        ValueError: If the stage is unsupported.
    """
    if stage == "sft":
        return SFT_PROMPT_TEMPLATE.format(schema = OUTPUT_SCHEMA_TEXT, **payload)
    if stage == "dpo":
        return DPO_PROMPT_TEMPLATE.format(schema = OUTPUT_SCHEMA_TEXT, **payload)
    if stage == "grpo":
        return GRPO_PROMPT_TEMPLATE.format(**payload)
    raise ValueError(f"Unsupported generation stage: {stage}")
