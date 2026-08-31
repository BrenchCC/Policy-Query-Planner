import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

EMBEDDING_TEXT_TEMPLATE = "{title}\n\n{text}"
PLANNER_SYSTEM_PROMPT = (
    "You are a retrieval query planner. Return valid JSON only. Do not answer the user's question. "
    "Preserve named entities, dates, quantities, relationships, and eligibility constraints."
)

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

The chosen plan is correct. Produce one plausible but meaningfully worse plan with the requested error type. The rejected plan must remain valid JSON and look superficially reasonable, but it must reduce retrieval quality. Do not make it nonsensical, do not answer the question, and do not violate the JSON schema.

Single-hop error types:
- unresolved_reference: keep an ambiguous pronoun or elliptical reference unresolved.
- entity_omission: omit the central named entity, policy, or relationship.
- constraint_omission: remove one eligibility-changing date, amount, duration, status, or condition.
- overly_broad: replace the focused query with a generic topic query.
- wrong_context: import a plausible but incorrect entity or constraint from the history or scenario.

Multi-hop planning error types:
- step_omission: remove one necessary retrieval step.
- broken_dependency: remove the link between a dependent query and its prerequisite.
- redundant_step: repeat an earlier retrieval step instead of seeking new evidence.
- overly_broad_step: replace one focused hop with a generic topic query.
- relation_omission: omit the relation that makes one hop retrieve the needed fact.

Hard rules:
1. Return exactly one JSON object and no Markdown.
2. Preserve valid sequential query ids and valid dependency placeholders.
3. The rejected query must differ from the chosen query.
4. Apply only the requested primary error type.
5. Never include the gold answer.
6. For single_hop, return one query. For multi_hop, keep the output superficially plausible while damaging the requested planning property.

Output shape:
{schema}

Input:
User input:
{user_input}

Chosen plan:
{chosen}

Required error type: {error_type}
Task type: {task_type}
Expected chosen hop count: {hop_count}
"""

GRPO_PROMPT_TEMPLATE = """You are creating one grounded synthetic multi-hop example for a public-policy retrieval planner.

Task:
Use the indexed policy evidence to write a new scenario and question that genuinely require two to four retrieval steps. Then provide the reference query plan and private reward metadata. Do not merely split the original question into several paraphrases.

Hard rules:
1. Return exactly one JSON object and no Markdown.
2. Create a new question, not a paraphrase of the source question.
3. The plan must contain two to four non-redundant queries with sequential ids.
4. At least one later query must depend on an earlier answer.
5. Dependencies may only refer to earlier queries and must use placeholders such as {{{{q1.answer}}}}.
6. The plan must not reveal the final reference answer.
7. hop_answers must contain one short evidence-grounded answer for every query.
8. hop_evidence_indices must contain one indexed evidence number for every query.
9. Each hop answer must be directly supported by its selected evidence.
10. Different hops must use different evidence indices and different policy chunks.
11. reference_answer must be a short answer directly supported by the selected evidence.
12. Variant {variant_index} should use a different reasoning chain when another variant is requested.

Output shape:
{{"scenario":"...","question":"...","plan":{{"queries":[{{"id":"q1","query":"...","depends_on":[]}},{{"id":"q2","query":"... {{{{q1.answer}}}} ...","depends_on":["q1"]}}]}},"hop_answers":["...","..."],"hop_evidence_indices":[0,1],"reference_answer":"..."}}

Source material for inspiration:
Policy title: {title}
Original scenario: {scenario}
Original question: {question}

Indexed policy evidence for teacher use only:
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
