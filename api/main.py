from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.schemas import AnalyzeRequest, AnalyzeResponse, ClauseAnalysis
from api.risk_logic import RiskChecker
from models.predictor import ContractClassifier

# Initialize FastAPI
app = FastAPI(
    title="Contract Risk Analysis API",
    description="API for contract risk analysis using NLP",
    version="1.0.0"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model & risk checker (load once at startup)
classifier = ContractClassifier(model_path="models/save_weights")
risk_checker = RiskChecker()


def split_clauses(text: str) -> list:
    """Split contract text into clauses (by newline or period)."""
    # Simple splitting - can be upgraded with NLP sentence tokenizer
    clauses = [c.strip() for c in text.split('\n') if c.strip()]
    if len(clauses) == 1:
        # If no newlines, split by period
        clauses = [c.strip() + '.' for c in text.split('.') if c.strip()]
    return clauses


@app.get("/")
def health_check():
    """Health check endpoint."""
    return {"status": "ok", "message": "Contract Risk Analysis API is running"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze_contract(request: AnalyzeRequest):
    """
    POST /analyze - Analyze contract text
    
    Input: JSON with contract_text
    Output: JSON containing Clause Text, Label, Risk Level, Warning Message
    """
    try:
        contract_text = request.contract_text
        
        if not contract_text or len(contract_text.strip()) == 0:
            raise HTTPException(status_code=400, detail="Contract text cannot be empty")
        
        # Split into clauses
        clauses = split_clauses(contract_text)
        
        results = []
        for clause in clauses:
            # 1. Classify with Ivan's ContractClassifier
            prediction = classifier.predict(clause)
            label = prediction["label"]
            confidence = prediction["confidence"]
            
            # 2. Check risk with RiskChecker
            risk_level, warning_message = risk_checker.check_risk(clause, label)
            
            # 3. Build response
            results.append(ClauseAnalysis(
                clause_text=clause,
                label=label,
                confidence=confidence,
                risk_level=risk_level,
                warning_message=warning_message
            ))
        
        return AnalyzeResponse(
            total_clauses=len(results),
            results=results
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Run server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)