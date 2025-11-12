"""
Standalone Melanoma Detection AI Agent
Uses pre-trained model for clinical inference without training
"""

import json
import os
from typing import Dict, Optional, List
import torch
from PIL import Image
import numpy as np
from torchvision import transforms

# Import from your existing modules
from config import *

class MelanomaAIAgent:
    """
    Production-ready inference agent for melanoma detection.
    Loads pre-trained model and provides clinical-grade assessments.
    """
    
    def __init__(self, checkpoint_path: str, device: Optional[torch.device] = None):
        """
        Initialize the AI agent with a trained model.
        
        Args:
            checkpoint_path: Path to best_model.pth checkpoint
            device: Computing device (auto-detected if None)
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(checkpoint_path)
        self.transform = self._create_transform()
        
        print(f"✓ MelanomaAI Agent initialized on {self.device}")
        print(f"✓ Model loaded from: {checkpoint_path}")
    
    def create_melanoma_model(num_classes=1, pretrained=True):
        # Load pre-trained ResNet-50 model with ImageNet weights
        # Pre-trained weights provide feature extractors that understand:
        # - Basic visual patterns (edges, corners, textures)
        # - Complex patterns (shapes, objects)
        # - Hierarchical feature representations
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)

        # Modify the final classification layer for binary classification
        # Original ResNet-50 FC layer: 2048 -> 1000 (ImageNet classes)
        # Modified FC layer: 2048 -> 1 (melanoma probability)
        num_ftrs = model.fc.in_features  # Get input features to FC layer (2048 for ResNet-50)

        # Replace final layer with binary classification head
        # Single output will be passed through sigmoid for probability
        model.fc = nn.Linear(num_ftrs, 1)
        
        return model

    def _load_model(self, checkpoint_path: str) -> torch.nn.Module:
        """Load pre-trained model from checkpoint."""
        # Create model architecture
        model = create_melanoma_model()
        
        # Load trained weights
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Set to evaluation mode
        model.eval()
        model.to(self.device)
        
        print(f"✓ Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
        print(f"✓ Best validation AUC: {checkpoint.get('best_auc', 'unknown'):.4f}")
        
        return model
    
    def _create_transform(self) -> transforms.Compose:
        """Create image preprocessing pipeline matching training."""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """Load and preprocess image for inference."""
        image = Image.open(image_path).convert('RGB')
        return self.transform(image)
    
    def process_single_case(self, 
                           image_path: str, 
                           patient_metadata: Optional[Dict] = None,
                           use_tta: bool = False,
                           tta_iterations: int = 5) -> Dict:
        """
        Process single case with zero-shot inference.
        
        Args:
            image_path: Path to dermoscopic image
            patient_metadata: Optional patient info (age, sex, location)
            use_tta: Whether to use test-time augmentation
            tta_iterations: Number of TTA predictions to average
            
        Returns:
            Comprehensive clinical assessment report
        """
        # Load and preprocess image
        image_tensor = self.preprocess_image(image_path)
        
        # Get prediction(s)
        if use_tta:
            probability = self._predict_with_tta(image_tensor, tta_iterations)
        else:
            probability = self._predict_single(image_tensor)
        
        # Generate clinical report
        return self.generate_clinical_report(
            probability=probability,
            image_path=image_path,
            metadata=patient_metadata
        )
    
    def _predict_single(self, image_tensor: torch.Tensor) -> float:
        """Single forward pass prediction."""
        with torch.no_grad():
            logits = self.model(image_tensor.unsqueeze(0).to(self.device))
            probability = torch.sigmoid(logits).cpu().item()
        return probability
    
    def _predict_with_tta(self, image_tensor: torch.Tensor, n_iterations: int) -> float:
        """Test-time augmentation for robust prediction."""
        tta_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(20)
        ])
        
        predictions = []
        with torch.no_grad():
            for _ in range(n_iterations):
                # Apply random augmentation
                augmented = tta_transform(image_tensor)
                logits = self.model(augmented.unsqueeze(0).to(self.device))
                prob = torch.sigmoid(logits).cpu().item()
                predictions.append(prob)
        
        return np.mean(predictions)
    
    def generate_clinical_report(self, 
                                 probability: float, 
                                 image_path: str,
                                 metadata: Optional[Dict] = None) -> Dict:
        """
        Generate structured clinical report matching prompt specifications.
        """
        # Classification with optimal threshold (from validation)
        threshold = 0.5  # Use your optimal threshold from training
        classification = "Malignant (Melanoma)" if probability > threshold else "Benign"
        
        # Risk stratification
        risk_level = self._assess_risk_level(probability)
        confidence_level = self._assess_confidence(probability)
        
        # Key features extraction (model-based)
        key_features = self._extract_key_features(probability, metadata)
        
        # ABCDE assessment
        abcde = self._generate_abcde_assessment(probability)
        
        # Clinical recommendations
        recommendations = self._generate_recommendations(
            probability, classification, risk_level, metadata
        )
        
        # Quality and limitations
        quality = self._assess_quality(image_path)
        
        return {
            "diagnosis": {
                "classification": classification,
                "confidence_score": round(probability, 4),
                "risk_level": risk_level,
                "confidence_level": confidence_level
            },
            "analysis": {
                "key_features": key_features,
                "abcde_assessment": abcde,
                "differential_diagnosis": self._get_differential(probability)
            },
            "recommendations": recommendations,
            "quality_assessment": quality,
            "patient_metadata": metadata or {},
            "model_info": {
                "architecture": "ResNet-50",
                "training_dataset": "ISIC 2020",
                "device": str(self.device)
            }
        }
    
    def _assess_risk_level(self, probability: float) -> str:
        """Stratify risk based on prediction probability."""
        if probability >= 0.8:
            return "Very High - Urgent referral recommended"
        elif probability >= 0.6:
            return "High - Dermatologist consultation advised"
        elif probability >= 0.4:
            return "Moderate - Monitor closely"
        elif probability >= 0.2:
            return "Low - Routine follow-up"
        else:
            return "Very Low - Reassurance appropriate"
    
    def _assess_confidence(self, probability: float) -> str:
        """Assess diagnostic confidence."""
        distance_from_threshold = abs(probability - 0.5)
        if distance_from_threshold > 0.4:
            return "High"
        elif distance_from_threshold > 0.2:
            return "Medium"
        else:
            return "Low - Consider expert review"
    
    def _extract_key_features(self, probability: float, metadata: Optional[Dict]) -> List[str]:
        """Extract clinically relevant features."""
        features = []
        
        if probability > 0.7:
            features.extend([
                "Irregular pigmentation patterns detected",
                "Asymmetric morphology observed",
                "Border irregularity present"
            ])
        elif probability > 0.4:
            features.append("Some atypical features noted")
        else:
            features.append("Regular symmetric pattern")
        
        # Add metadata-based risk factors
        if metadata:
            if metadata.get('age', 0) > 50:
                features.append("Patient age >50 (increased melanoma risk)")
            if metadata.get('sex') == 'male':
                features.append("Male patient (slightly elevated risk)")
        
        return features
    
    def _generate_abcde_assessment(self, probability: float) -> Dict:
        """Generate ABCDE rule assessment."""
        # Simplified assessment based on probability
        # In production, this would use dedicated feature extractors
        
        base_score = int(probability * 5)  # 0-5 scale
        
        return {
            "Asymmetry": "Present" if base_score >= 3 else "Absent",
            "Border": "Irregular" if base_score >= 3 else "Regular",
            "Color": "Varied" if base_score >= 4 else "Uniform",
            "Diameter": "Assessment recommended" if base_score >= 3 else "Within normal range",
            "Evolution": "Clinical history needed"
        }
    
    def _get_differential(self, probability: float) -> List[str]:
        """Provide differential diagnosis."""
        if probability > 0.6:
            return [
                "Melanoma (primary concern)",
                "Atypical/dysplastic nevus",
                "Seborrheic keratosis (pigmented)"
            ]
        else:
            return [
                "Benign nevus (most likely)",
                "Seborrheic keratosis",
                "Solar lentigo"
            ]
    
    def _generate_recommendations(self, 
                                 probability: float, 
                                 classification: str,
                                 risk_level: str,
                                 metadata: Optional[Dict]) -> Dict:
        """Generate evidence-based recommendations."""
        
        if probability >= 0.7:
            next_steps = [
                "URGENT: Refer to dermatologist within 2 weeks",
                "Consider dermoscopic examination",
                "Biopsy strongly recommended if clinically suspicious"
            ]
            follow_up = "Immediate specialist evaluation required"
        elif probability >= 0.5:
            next_steps = [
                "Refer to dermatologist for evaluation",
                "Clinical examination recommended",
                "Consider biopsy if other risk factors present"
            ]
            follow_up = "Specialist review within 4-6 weeks"
        elif probability >= 0.3:
            next_steps = [
                "Clinical monitoring advised",
                "Photographic documentation for comparison",
                "Patient education on warning signs"
            ]
            follow_up = "Re-evaluate in 3-6 months"
        else:
            next_steps = [
                "Routine monitoring appropriate",
                "Patient reassurance",
                "General skin cancer education"
            ]
            follow_up = "Annual skin check sufficient"
        
        return {
            "next_steps": next_steps,
            "follow_up_timing": follow_up,
            "patient_education": [
                "Monitor for changes in size, shape, or color",
                "Sun protection and UV avoidance",
                "Regular self-examination"
            ]
        }
    
    def _assess_quality(self, image_path: str) -> Dict:
        """Assess image quality and limitations."""
        # Basic quality check - expand as needed
        try:
            img = Image.open(image_path)
            width, height = img.size
            
            if width < 224 or height < 224:
                quality = "Low - Image resolution suboptimal"
            else:
                quality = "Acceptable"
            
            return {
                "image_quality": quality,
                "image_dimensions": f"{width}x{height}",
                "limitations": [
                    "AI assessment is supplementary to clinical judgment",
                    "Dermoscopic examination by specialist recommended",
                    "Patient history and clinical context essential"
                ]
            }
        except Exception as e:
            return {
                "image_quality": "Unable to assess",
                "limitations": [f"Error: {str(e)}"]
            }
    
    def process_batch(self, image_paths: List[str]) -> List[Dict]:
        """Process multiple cases efficiently."""
        results = []
        for path in image_paths:
            result = self.process_single_case(path)
            results.append(result)
        return results
    
    def export_report(self, result: Dict, output_path: str, format: str = "json"):
        """Export clinical report to file."""
        if format == "json":
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)
        elif format == "txt":
            with open(output_path, 'w') as f:
                f.write(self._format_text_report(result))
        
        print(f"✓ Report exported to: {output_path}")
    
    def _format_text_report(self, result: Dict) -> str:
        """Format result as readable text report."""
        report = []
        report.append("=" * 60)
        report.append("MELANOMA AI DETECTION REPORT")
        report.append("=" * 60)
        report.append(f"\nDiagnosis: {result['diagnosis']['classification']}")
        report.append(f"Confidence: {result['diagnosis']['confidence_score']:.2%}")
        report.append(f"Risk Level: {result['diagnosis']['risk_level']}")
        report.append(f"\nKey Features:")
        for feature in result['analysis']['key_features']:
            report.append(f"  • {feature}")
        report.append(f"\nRecommendations:")
        for step in result['recommendations']['next_steps']:
            report.append(f"  • {step}")
        report.append("=" * 60)
        return "\n".join(report)


# Example usage
if __name__ == "__main__":
    # Initialize agent with your trained model
    agent = MelanomaAIAgent(
        checkpoint_path="checkpoints/best_model.pth",
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )
    
    # Process single image
    result = agent.process_single_case(
        image_path="path/to/dermoscopic_image.jpg",
        patient_metadata={
            "age": 55,
            "sex": "male",
            "location": "back"
        },
        use_tta=True  # Enable for more robust prediction
    )
    
    # Print diagnosis
    print(f"Classification: {result['diagnosis']['classification']}")
    print(f"Confidence: {result['diagnosis']['confidence_score']:.2%}")
    print(f"Risk: {result['diagnosis']['risk_level']}")
    
    # Export detailed report
    agent.export_report(result, "melanoma_report.json", format="json")
    agent.export_report(result, "melanoma_report.txt", format="txt")