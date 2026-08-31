import os
import sys

# Add project root to Python path
sys.path.append(os.getcwd())

from data_preprocess.prompts import PLANNER_SYSTEM_PROMPT


def test_planner_system_prompt_is_owned_by_prompts_module() -> None:
    """Keep the planner LLM system prompt with the other prompt templates."""
    assert "retrieval query planner" in PLANNER_SYSTEM_PROMPT
    assert "Return valid JSON only" in PLANNER_SYSTEM_PROMPT
