# AI Agent System Architecture

## End-to-End Pipeline Overview

```
Input (x) -> Preprocessing -> Feature Extraction -> Classification -> Clinical Reasoning -> Output (y)
```

## Component Mapping to Current Codebase

### 1. Input Processing
- **File**: `dataset.py`- `MelanomaDataset` class
- **Function**: Image loading, validation, normalization
- **AI Agent Role**: Standardized input acceptance

### 2. Feature Extraction
- **File**: `models.py` - ResNet-50 architecture
- **Function**: Deep feature extraction from dermoscopy images
- **AI Agent Role**: Pattern recognition and feature analysis

### 3. Classification
- **File**: `main.py` - Training and inference pipeline
- **Function**: Binary classification with confidence scores
- **AI Agent Role**: Primary diagnostic decision making

### 4. Clinical Reasoning
- **File**: `evaluation.py` - Metrics and threshold optimization
- **Function**: Convert model outputs to clinical language
- **AI Agent Role**: Evidence-based reasoning and uncertainty quantification

### 5. Output Generation
- **Integration Point**: New module needed for structured clinical reports
- **Function** Convert model outputs to clinical language
- **AI Agent Role**: Human-interpretable diagnostic communication

## Prompt Integration Strategy

1. **Wrap existing pipeline** in natural language interface
2. **Add clinical reasoning layer** for ABCDE assessment
3. **Implement structure output** generation from model predictions
4. **Include uncertainty quantification** for clinical safety