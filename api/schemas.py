from pydantic import BaseModel
from typing import List, Optional

# Request Schema
class AnalyzeRequest(BaseModel):
    contract_text: str  # Contract text to analyze

# Response Schema for each clause
class ClauseAnalysis(BaseModel):
    clause_text: str       # Clause content
    label: str             # Predicted label (Terminations, Confidentiality, etc.)
    confidence: float      # Confidence score (0.0 - 1.0)
    risk_level: str        # "LOW", "MEDIUM", "HIGH"
    warning_message: str   # Risk warning message

# Aggregated Response
class AnalyzeResponse(BaseModel):
    total_clauses: int
    results: List[ClauseAnalysis]