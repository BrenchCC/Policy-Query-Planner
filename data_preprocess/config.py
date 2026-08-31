import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
RAW_ROOT = DATA_ROOT / "raw"
INTERIM_ROOT = DATA_ROOT / "interim"
PROCESSED_ROOT = DATA_ROOT / "processed"
REPORT_ROOT = DATA_ROOT / "reports"
REQUEST_ROOT = DATA_ROOT / "generation_requests"
RESPONSE_ROOT = DATA_ROOT / "generation_responses"

CONDITIONALQA_RAW_ROOT = RAW_ROOT / "conditionalqa"
QRECC_RAW_ROOT = RAW_ROOT / "qrecc"
MUSIQUE_RAW_ROOT = RAW_ROOT / "musique"
ARK_RAW_ROOT = RAW_ROOT / "ark_helper"

CONDITIONALQA_REVISION = "77bd295952daf415548b3244db10880d3d55cfe0"
QRECC_REVISION = "cf44d03cb6676f7414471cec509d4a6c6858b0d3"
MUSIQUE_REVISION = "922ac98f19a201998dbdae6d7f2887a5258dbdeb"
ARK_SCRIPT_REVISION = "158bcbb94d50db6ac92e271fb96cae326dbaf642"

CONDITIONALQA_FILES = [
    "documents.json",
    "train.json",
    "dev.json",
    "test_no_answer.json"
]
QRECC_ARCHIVE_NAME = "qrecc_data.zip"
QRECC_ARCHIVE_SHA256 = "6ed52a0b0e495f752424b15742119a5f8f33ca7a029c07122be1b849de19a308"
MUSIQUE_ARCHIVE_NAME = "musique_v1.0.zip"
MUSIQUE_DRIVE_ID = "1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h"

EXPECTED_CONDITIONALQA_COUNTS = {
    "documents": 652,
    "train": 2338,
    "dev": 285,
    "test_no_answer": 804
}
EXPECTED_QRECC_COUNTS = {"train": 63501, "test": 16451}
TARGET_COUNTS = {
    "sft": 20000,
    "sft_multihop_cold_start": 2000,
    "dpo": 5000,
    "grpo": 5000
}
MULTIHOP_COLD_START_HOP_QUOTAS = {2: 1000, 3: 600, 4: 400}
DPO_SINGLE_SOURCE_QUOTAS = {"qrecc": 2000, "conditionalqa": 500}
DPO_MULTIHOP_HOP_QUOTAS = {2: 1250, 3: 1000, 4: 250}
GRPO_SINGLE_COUNT = 1000
GRPO_MULTIHOP_COUNT = 4000
GRPO_HOP_MINIMUM = 500

RANDOM_SEED = 42
MAX_QUERY_COUNT = 4
MAX_POLICY_CHUNK_CHARS = 1600
MIN_POLICY_CHUNK_CHARS = 120
EVIDENCE_MAPPING_THRESHOLD = 0.99

PLANNER_SYSTEM_PROMPT = (
    "You are a retrieval query planner. Return valid JSON only. Do not answer the user's question. "
    "Preserve named entities, dates, quantities, relationships, and eligibility constraints."
)

LLM_ENDPOINT = os.environ.get("LLM_ENDPOINT", "")
