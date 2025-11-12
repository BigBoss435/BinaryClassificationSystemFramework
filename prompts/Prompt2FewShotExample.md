# Melanoma Detection AI Agent - Few-Shot Prompt

## System Description
You are a medical AI agent specialized in melanoma classification. Learn from the following examples to understand the diagnostic patterns and reasoning process.

### Training Examples (Known X,y pairs)
**Input (X1)**:
- Image: Regular, symmetric brown lesion, uniform color
- Patient: 28-year-old male, chest location
- History: Stable for 2+ years

**Output (y1)**:
```json
{
  "diagnosis": {
    "classification": "Benign",
    "confidence_score": 0.92,
    "risk_level": "Low"
  },
  "analysis": {
    "key_features": [
      "Symmetric appearance",
      "Regular borders",
      "Uniform brown color",
      "Stable history"
    ],
    "abcde_assessment": {
      "asymmetry": "Absent",
      "border": "Regular",
      "color": "Uniform",
      "diameter": "<6mm",
      "evolution": "Stable"
    }
  },
  "recommendations": {
    "urgency": "Routine",
    "next_steps": "Continue self-monitoring",
    "follow_up": "Annual skin check"
  }
}
```

### Example 2: Suspicious Melanoma
**Input (X2)**:
- Image: Assymetric lesion with irregular borders, multiple colors
- Patient: 52-year-old female, shoulder
- History: Recent growth and color change over 3 months

**Output (y2)**:
```json
{
  "diagnosis": {
    "classification": "Malignant",
    "confidence_score": 0.87,
    "risk_level": "High"
  },
  "analysis": {
    "key_features": [
      "Marked asymmetry",
      "Irregular, notched borders",
      "Multiple colors (brown, black, red)",
      "Recent evolution reported"
    ],
    "abcde_assessment": {
      "asymmetry": "Present",
      "border": "Irregular",
      "color": "Variegated",
      "diameter": ">6mm",
      "evolution": "Changed"
    }
  },
  "recommendations": {
    "urgency": "Immediate",
    "next_steps": "Urgent dermatologist referral for biopsy",
    "follow_up": "Within 1-2 weeks"
  }
}
```

### Example 3: Uncertain Case - Atypical Nevus
**Input (X3)**:
- Image: Moderately asymmetric lesion, some border irregularity
- Patient: 35-year-old male, back
- History: Slowly growing over 1 year

**Output (y3)**:
```json
{
  "diagnosis": {
    "classification": "Benign",
    "confidence_score": 0.73,
    "risk_level": "Moderate"
  },
  "analysis": {
    "key_features": [
      "Mild asymmetry",
      "Slightly irregular borders",
      "Predominantly uniform color with minor variation",
      "Slow growth pattern"
    ],
    "abcde_assessment": {
      "asymmetry": "Mild",
      "border": "Slightly irregular",
      "color": "Mostly uniform",
      "diameter": "6-8mm",
      "evolution": "Slow growth"
    }
  },
  "recommendations": {
    "urgency": "Moderate",
    "next_steps": "Dermatologist evaluation recommended",
    "follow_up": "3-month monitoring if benign confirmed",
    "note": "Consider dermoscopy or biopsy if clinical suspicion remains"
  }
}
```

## New Case for Analysis

**Input (X_new)**:
- **Image**: [dermoscopy_lesion_unknown.jpg]
- **Patient Metadata**:
    - Age: 41 years
    - Sex: Male
    - Location: Left arm
    - History: Patient noticed darkening of existing mole over past 2 months
    - Family history: Mother had melanoma at age 60

**Task**: Based on the patterns learned from the examples above, provide complete diagnostic assessment for this new case following the same structured format and reasoning approach.

## Analysis Guidelines
1. **Pattern Recognition**: Compare visual features with the training examples
2. **Risk Stratification**: Consider patient factors (age, family history, location)
3. **Clinical Context**: Weight the significance of reported changes
4. **Diagnostic Confidence**: Acknowledge uncertainty when present
5. **Clinical Reasoning**: Explain the decision-making process
6. **Safety First**: Be on the side of caution for patient safety

## Expected Response
Provide the same structured JSON format as shown in the examples, with detailed reasoning for each component of your assessment.