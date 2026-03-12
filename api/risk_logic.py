from typing import Dict, List, Tuple

class RiskChecker:
    """
    Rule-based Risk Engine: Check risk keywords for each Label.
    """
    
    # Risk keywords for each label
    RISK_KEYWORDS: Dict[str, Dict[str, List[str]]] = {
        "Terminations": {
            "high": ["immediate termination", "without cause", "at will", "without notice"],
            "medium": ["30 days notice", "notice period", "breach", "default"],
            "low": ["mutual agreement", "90 days notice", "cure period"]
        },
        "Confidentiality": {
            "high": ["unlimited duration", "perpetual", "no exceptions", "all information"],
            "medium": ["5 years", "proprietary information", "trade secrets"],
            "low": ["public information", "prior written consent", "reasonable exceptions"]
        },
        "Indemnifications": {
            "high": ["unlimited liability", "all damages", "gross negligence", "sole discretion"],
            "medium": ["reasonable costs", "third party claims", "defend and hold harmless"],
            "low": ["cap on liability", "mutual indemnification", "insurance coverage"]
        },
        "Assignments": {
            "high": ["non-assignable", "without consent", "sole discretion"],
            "medium": ["prior written consent", "affiliate", "successor"],
            "low": ["freely assignable", "mutual consent", "notice required"]
        },
        "Governing Laws": {
            "high": ["foreign jurisdiction", "arbitration mandatory"],
            "medium": ["specific state law", "exclusive jurisdiction"],
            "low": ["mutual agreement", "mediation first"]
        }
    }
    
    WARNING_MESSAGES: Dict[str, Dict[str, str]] = {
        "Terminations": {
            "high": "CRITICAL: Clause allows termination without prior notice!",
            "medium": "CAUTION: Review notice period and breach conditions carefully.",
            "low": "OK: Termination clause is reasonable."
        },
        "Confidentiality": {
            "high": "CRITICAL: Confidentiality obligation is too broad or indefinite!",
            "medium": "CAUTION: Review scope and duration of confidentiality.",
            "low": "OK: Confidentiality clause is reasonable."
        },
        "Indemnifications": {
            "high": "CRITICAL: Unlimited indemnification liability!",
            "medium": "CAUTION: Review scope of indemnification.",
            "low": "OK: Indemnification clause is balanced."
        },
        "Assignments": {
            "high": "CRITICAL: Strict assignment restrictions!",
            "medium": "CAUTION: Consent required before assignment.",
            "low": "OK: Assignment clause is flexible."
        },
        "Governing Laws": {
            "high": "CRITICAL: Unfavorable or foreign governing law!",
            "medium": "CAUTION: Review jurisdiction carefully.",
            "low": "OK: Governing law clause is reasonable."
        }
    }
    
    def check_risk(self, text: str, label: str) -> Tuple[str, str]:
        """
        Check risk level based on keywords.
        
        Args:
            text: Clause content
            label: Classified label
            
        Returns:
            Tuple[risk_level, warning_message]
        """
        text_lower = text.lower()
        
        if label not in self.RISK_KEYWORDS:
            return "UNKNOWN", "Label not recognized in the system."
        
        keywords = self.RISK_KEYWORDS[label]
        
        # Check HIGH risk first
        for keyword in keywords.get("high", []):
            if keyword in text_lower:
                return "HIGH", self.WARNING_MESSAGES[label]["high"]
        
        # Check MEDIUM risk
        for keyword in keywords.get("medium", []):
            if keyword in text_lower:
                return "MEDIUM", self.WARNING_MESSAGES[label]["medium"]
        
        # Default to LOW
        return "LOW", self.WARNING_MESSAGES[label]["low"]