# Melanoma Detection AI Agent Prompts

This directory contains prompt examples for interacting with the melanoma classification system as an end-to-end AI agent.

## System Overview

Our AI agent transform dermoscopy images into clinical diagnostic assessments through:

1. **Image Analysis Pipeline**
    - ResNet-50 based feature extraction
    - Multi-scale analysis and augmentation
    - Quality assessment and artifact detection

2. **Clinical Reasoning Engine**
    - ABCDE criteria evaluation
    - Risk stratification algorithms
    - Evidence-based decision making

3. **Report Generation System**
    - Structured clinical outputs
    - Confidence quantification
    - Actionable recommendations

## Prompt Files

- `Prompt1ZeroShotExample.md`: Single image analysis without examples
- `Prompt2FewShotExample.md`: Learning from multiple examples before new prediction
- `SystemArchitecture.md`: Technical details of the AI agent implementation

## Integration with Current System

These prompts can be integrated with our existing codebase by:

```python
# Example integration point in main.py
def ai_agent_interface(image_path, patient_metadata=None, few_shot_examples=None):
    """
    End-to-end AI agent interface matching prompt specifications
    """
    # Load and preprocess image
    image = load_and_preprocess_image(image_path)
    
    # Run through model pipeline
    model_output = model.predict(image)
    
    # Generate structured clinical report
    clinical_report = generate_clinical_report(
        model_output, 
        patient_metadata, 
        few_shot_examples
    )
    
    return clinical_report
```

## Usage Examples

See individual files for detailed examples and expected input/output formats.