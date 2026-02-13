import torch
import os
import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

# --- SETUP LOGGING ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ContractPredictor")

class ContractClassifier:
    """
    Wrapper class for Legal-BERT Inference.
    Loads the model once and provides a method to predict contract clauses.
    """
    
    def __init__(self, model_path: str = "models/saved_weights"):
        """
        Initialize the model and tokenizer from saved artifacts.
        
        Args:
            model_path (str): Path to the directory containing 'pytorch_model.bin' and 'config.json'.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading model from '{model_path}' using device: {self.device}...")

        if not os.path.exists(model_path):
            logger.error(f"Model path not found: {model_path}")
            raise FileNotFoundError(f"Directory '{model_path}' does not exist.")

        try:
            # Load Tokenizer & Model
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            
            # Move model to GPU if available for faster inference
            self.model.to(self.device)
            
            # Set model to evaluation mode (Deactivates Dropout layers)
            self.model.eval()
            
            # Extract Label Mapping from config (e.g., {0: "Termination", ...})
            self.id2label = self.model.config.id2label
            logger.info("Model loaded successfully!")
            
        except Exception as e:
            logger.critical(f"Failed to load model: {str(e)}")
            raise e

    def predict(self, text: str):
        """
        Analyze a single text clause and return the predicted label.
        
        Args:
            text (str): The contract clause text to analyze.
            
        Returns:
            dict: {
                "label": str (e.g., "Termination"),
                "confidence": float (e.g., 0.9852),
                "risk_score": float (reserved for future logic)
            }
        """
        if not text or not isinstance(text, str):
            return {"label": "Unknown", "confidence": 0.0}

        # 1. Preprocessing (Tokenization)
        # Convert text to numbers, truncate to max length model can handle
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=256, 
            padding=True
        )
        
        # Move inputs to the same device as model
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 2. Inference (No Gradient Calculation)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits # Raw output scores
            
            # 3. Post-processing (Softmax)
            # Convert raw logits to probabilities (0.0 to 1.0)
            probs = F.softmax(logits, dim=-1)
            
            # Find the index with the highest probability
            pred_idx = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][pred_idx].item()
            
            predicted_label = self.id2label[pred_idx]

        return {
            "label": predicted_label,
            "confidence": round(confidence, 4)
        }

# --- QUICK TEST (For debugging) ---
if __name__ == "__main__":
    # Test path assumption
    test_path = "models/saved_weights" if os.path.exists("models/saved_weights") else "saved_weights"
    
    try:
        classifier = ContractClassifier(model_path=test_path)
        sample_text = "The employment may be terminated by either party upon providing 30 days written notice."
        
        print("-" * 30)
        print(f"Input: {sample_text}")
        result = classifier.predict(sample_text)
        print(f"Prediction: {result}")
        print("-" * 30)
    except Exception as e:
        print(f"Error during test: {e}")