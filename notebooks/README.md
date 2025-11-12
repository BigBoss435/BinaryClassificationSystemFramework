# Melanoma Detection AI - Google Colab Demo

## Lab 1.5: End-to-End AI Solution with Gemini API

This directory contains the complete implementation of the Melanoma Detection AI Agent as a Google Colab notebook, demonstrating integration with Google's Gemini API for enhanced clinical reasoning.

---

## 📁 Files

- **`Melanoma_Detection_Gemini_Demo.ipynb`** - Main Colab notebook with complete implementation

---

## 🚀 Quick Start

### Step 1: Open in Google Colab

1. Upload `Melanoma_Detection_Gemini_Demo.ipynb` to Google Drive
2. Right-click → Open with → Google Colaboratory
3. Or use this link format: `https://colab.research.google.com/github/BigBoss435/BinaryClassificationSystemFramework/blob/main/notebooks/Melanoma_Detection_Gemini_Demo.ipynb`

### Step 2: Get Gemini API Key

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Copy the key (you'll use it in Step 3)

### Step 3: Configure Secrets in Colab

1. In Colab, click the **🔑 Secrets** icon in the left sidebar
2. Click **+ Add new secret**
3. Name: `GEMINI_KEY`
4. Value: Paste your API key
5. Toggle on "Notebook access"

### Step 4: Run the Notebook

1. Go to **Runtime → Run all**
2. Or execute cells sequentially (Shift+Enter)
3. Watch the complete pipeline demonstration!

---

## 📊 What the Notebook Demonstrates

### 1. **Setup & Configuration**
- Secure API key management using Colab Secrets
- Library installation and initialization
- Model architecture setup

### 2. **Prompt Engineering (Lab 1.4 Integration)**
- **Zero-Shot Prompt**: Direct task description without examples
- **Few-Shot Prompt**: Learning from 3 training examples
- Both prompts from `/prompts/` directory

### 3. **End-to-End Pipeline**
```
Input Image → Preprocessing → CNN Inference → 
Gemini Reasoning → Clinical Report → Output
```

### 4. **Three Test Cases**
- **Case 1**: Low-risk benign nevus (P=0.15)
- **Case 2**: High-risk suspicious melanoma (P=0.85)
- **Case 3**: Moderate-risk atypical nevus (P=0.58)

### 5. **Results & Visualization**
- Risk distribution charts
- Prompt comparison (zero-shot vs few-shot)
- Exportable JSON reports

---

## 🔬 System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    INPUT LAYER                          │
│  Dermoscopic Image + Patient Metadata                   │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│              PREPROCESSING                              │
│  Resize (224x224) → Normalize → Tensor Conversion       │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│           DEEP LEARNING MODEL                           │
│  ResNet-50 → Feature Extraction → Binary Classification │
│  Output: P(melanoma) ∈ [0, 1]                          │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│         GEMINI API INTEGRATION                          │
│  ┌──────────────────┐  ┌─────────────────┐            │
│  │  Zero-Shot       │  │  Few-Shot       │            │
│  │  Reasoning       │  │  Learning       │            │
│  └──────────────────┘  └─────────────────┘            │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│           CLINICAL REPORT GENERATION                    │
│  • Diagnosis & Confidence                               │
│  • ABCDE Assessment                                     │
│  • Risk Stratification                                  │
│  • Clinical Recommendations                             │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│                   OUTPUT                                │
│  Structured JSON Report + Visualizations                │
└─────────────────────────────────────────────────────────┘
```

---

## 📝 Expected Outputs

### For Each Test Case:

**Console Output:**
```
🔵 STAGE 1: Data Understanding
  Case ID: CASE_002
  Patient: 52yo female
  Location: shoulder

🔵 STAGE 2: Deep Learning Inference
  Model Output: P(melanoma) = 0.850
  Classification: MALIGNANT

🔵 STAGE 3a: Gemini Reasoning (Zero-Shot)
  ✓ Zero-shot assessment complete

🔵 STAGE 3b: Gemini Reasoning (Few-Shot)
  ✓ Few-shot assessment complete

🔵 STAGE 4: Report Generation
  ✓ Clinical report generated

✅ Pipeline Complete!
```

**JSON Report Structure:**
```json
{
  "diagnosis": {
    "classification": "Malignant",
    "confidence_score": 0.87,
    "risk_level": "High"
  },
  "analysis": {
    "key_features": [...],
    "abcde_assessment": {...}
  },
  "recommendations": {
    "urgency": "Immediate",
    "next_steps": "Urgent dermatologist referral for biopsy",
    "follow_up": "Within 1-2 weeks"
  }
}
```

---

## 🎯 Learning Objectives Achieved

✅ **Task 1**: Created structured Google Colab notebook with clear documentation  
✅ **Task 2**: Configured Gemini API securely using Colab Secrets  
✅ **Task 3**: Integrated both prompt examples from Lab 1.4  
✅ **Task 4**: Demonstrated complete end-to-end pipeline with all stages  
✅ **Task 5**: Documented system with reflection and analysis  

---

## 🛠️ Customization Options

### Use Your Own Images

Replace the simulated probabilities with actual image processing:

```python
# Instead of:
probability = case['simulated_probability']

# Use:
image_tensor = preprocess_image('path/to/your/image.jpg')
probability = predict_melanoma(image_tensor, use_tta=True)
```

### Upload Trained Model

If you have a trained checkpoint:

```python
# Upload best_model.pth to Colab
from google.colab import files
uploaded = files.upload()

# Load model
checkpoint = torch.load('best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
```

### Modify Prompts

Edit the `ZERO_SHOT_PROMPT` and `FEW_SHOT_PROMPT` variables to customize:
- Clinical criteria emphasis
- Output format preferences
- Risk threshold adjustments
- Additional metadata integration

---

## 📚 References

- **Prompt Examples**: See `/prompts/Prompt1ZeroShotExample.md` and `Prompt2FewShotExample.md`
- **System Architecture**: See `/prompts/SystemArchitecture.md`
- **Inference Agent**: See `/prompts/inference_agent.py`

---

## ⚠️ Important Notes

### For Academic/Research Use Only

This system is a **demonstration project** for educational purposes. It is NOT validated for clinical use and should NOT be used for actual medical diagnosis.

### Model Training Required

The notebook uses simulated probabilities. For real deployment:
1. Train on validated datasets (e.g., ISIC 2019, HAM10000)
2. Validate with dermatologist ground truth
3. Conduct clinical trials
4. Obtain regulatory approval

### API Costs

Gemini API calls may incur costs depending on your usage tier. Monitor your usage at [Google AI Studio](https://makersuite.google.com/).

---

## 🤝 Contributing

Found an issue or have suggestions? 
- Open an issue on GitHub
- Submit a pull request
- Contact: [Your contact information]

---

## 📄 License

[Specify your license here]

---

**Created**: November 12, 2025  
**Course**: Advanced AI Systems Laboratory - Lab 1.5  
**Author**: [Your name]
