# Lab 1.5 Setup Guide

## Quick Start for Google Colab

### Prerequisites
- Google account
- Gemini API key (free tier available)
- GitHub repository access

---

## Step-by-Step Instructions

### 1️⃣ Get Your Gemini API Key

1. Visit: https://makersuite.google.com/app/apikey
2. Sign in with your Google account
3. Click **"Create API Key"**
4. Select a Google Cloud project (or create new)
5. Copy the generated API key
6. Keep it secure - don't share or commit to GitHub!

---

### 2️⃣ Open the Notebook in Colab

**Option A: From GitHub (Recommended)**
1. Go to Google Colab: https://colab.research.google.com/
2. Click **File → Open notebook**
3. Select **GitHub** tab
4. Enter repository: `BigBoss435/BinaryClassificationSystemFramework`
5. Open: `notebooks/Melanoma_Detection_Gemini_Demo.ipynb`

**Option B: Upload Directly**
1. Download `Melanoma_Detection_Gemini_Demo.ipynb` from GitHub
2. Go to: https://colab.research.google.com/
3. Click **File → Upload notebook**
4. Select the downloaded file

---

### 3️⃣ Configure API Key in Colab

**IMPORTANT: Use Colab Secrets (Not Hardcoding!)**

1. In the Colab notebook, look at the **left sidebar**
2. Click the **🔑 Secrets** icon (Key icon)
3. Click **"+ Add new secret"**
4. Fill in:
   - **Name**: `GEMINI_KEY` (exactly as shown)
   - **Value**: Paste your API key from Step 1
5. Toggle **ON** the "Notebook access" switch
6. Click **Save**

**Why use Secrets?**
- Keeps your API key secure
- Prevents accidental commits to GitHub
- Follows security best practices

---

### 4️⃣ Run the Notebook

**Option A: Run All Cells**
1. Click **Runtime → Run all** from the menu
2. Wait for all cells to execute (2-3 minutes)
3. Scroll through to see results

**Option B: Run Step-by-Step**
1. Click on the first code cell
2. Press **Shift + Enter** to execute
3. Read the output
4. Continue with next cell
5. Repeat until the end

---

### 5️⃣ What to Expect

#### Installation Phase (1-2 minutes)
```
Installing packages...
✓ All libraries imported successfully
✓ PyTorch version: 2.x.x
✓ CUDA available: True/False
```

#### API Configuration
```
✓ Gemini API configured successfully
✓ Initialized model: gemini-1.5-flash
```

#### Model Setup
```
✓ Model created and loaded on: cuda/cpu
✓ Total parameters: 25,557,032
✓ Preprocessing pipeline configured
```

#### Pipeline Execution (per case)
```
🔵 STAGE 1: Data Understanding
🔵 STAGE 2: Deep Learning Inference
🔵 STAGE 3a: Gemini Reasoning (Zero-Shot)
🔵 STAGE 3b: Gemini Reasoning (Few-Shot)
🔵 STAGE 4: Report Generation
✅ Pipeline Complete!
```

#### Visualizations
- Bar chart showing melanoma probabilities
- Pie chart with risk stratification
- Comparison tables between prompt types

#### Exported Results
- `melanoma_detection_results.json` (downloadable)

---

## Troubleshooting

### ❌ "Error loading API key"

**Problem**: Gemini API key not found

**Solutions**:
1. Check the secret name is exactly `GEMINI_KEY` (case-sensitive)
2. Ensure "Notebook access" toggle is ON
3. Refresh the page and try again
4. Re-enter the API key in Secrets

---

### ❌ "Module not found" errors

**Problem**: Libraries not installed

**Solution**:
```python
# Run this in a cell:
!pip install --upgrade google-generativeai torch torchvision pillow
```

---

### ❌ "API quota exceeded"

**Problem**: Too many API calls

**Solutions**:
1. Wait a few minutes (rate limiting)
2. Check your quota: https://makersuite.google.com/
3. Reduce number of test cases
4. Use caching for repeated calls

---

### ❌ "CUDA out of memory"

**Problem**: GPU memory exhausted

**Solutions**:
1. The notebook automatically falls back to CPU
2. Or add this at the top:
```python
device = torch.device('cpu')  # Force CPU usage
```

---

### ❌ JSON parsing errors

**Problem**: Gemini response not in expected format

**Solution**: The notebook has fallback handling - check the `raw_response` field in output

---

## Understanding the Output

### Model Predictions

| Probability Range | Classification | Risk Level | Action |
|------------------|----------------|------------|--------|
| 0.00 - 0.30 | Benign | Low | Routine monitoring |
| 0.30 - 0.70 | Uncertain | Moderate | Dermatologist evaluation |
| 0.70 - 1.00 | Malignant | High | Urgent biopsy |

### Report Structure

```json
{
  "case_id": "CASE_XXX",
  "timestamp": "2025-11-12T...",
  "stages": {
    "input": {...},              // Patient metadata
    "inference": {...},          // Model prediction
    "zeroshot_reasoning": {...}, // Prompt 1 results
    "fewshot_reasoning": {...}   // Prompt 2 results
  },
  "final_report": {...}          // Clinical assessment
}
```

---

## Testing Your Setup

### Quick Test (Before Running Full Notebook)

Add this cell at the beginning:

```python
# Test Gemini API connection
import google.generativeai as genai
from google.colab import userdata

try:
    GEMINI_KEY = userdata.get('GEMINI_KEY')
    genai.configure(api_key=GEMINI_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    response = model.generate_content("Say 'API working!' if you can read this.")
    print(response.text)
    print("✅ Gemini API is working correctly!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("Check your API key configuration in Secrets")
```

Expected output:
```
API working!
✅ Gemini API is working correctly!
```

---

## Advanced: Using Real Images

### Upload Your Own Dermoscopic Images

```python
from google.colab import files

# Upload images
uploaded = files.upload()

# Process uploaded image
for filename in uploaded.keys():
    print(f"Processing: {filename}")
    
    # Preprocess
    image_tensor = preprocess_image(filename)
    
    # Predict
    probability = predict_melanoma(image_tensor, use_tta=True)
    
    # Generate report
    result = generate_gemini_report(
        probability=probability,
        metadata={
            "age": 45,
            "sex": "female",
            "location": "back",
            "history": "Recent changes noted"
        }
    )
    
    print(json.dumps(result, indent=2))
```

---

## Submission Checklist for Lab 1.5

- [ ] Notebook runs successfully in Colab
- [ ] API key configured securely (in Secrets, not hardcoded)
- [ ] Both prompts (zero-shot and few-shot) demonstrated
- [ ] All three test cases processed
- [ ] Visualizations generated
- [ ] Results exported to JSON
- [ ] Reflection section completed
- [ ] Notebook saved to GitHub `/notebooks/` directory
- [ ] README documentation included

---

## Additional Resources

### Documentation
- **Gemini API Docs**: https://ai.google.dev/docs
- **PyTorch Tutorials**: https://pytorch.org/tutorials/
- **Colab Guide**: https://colab.research.google.com/notebooks/

### Datasets (for future work)
- **ISIC 2019**: https://challenge.isic-archive.com/
- **HAM10000**: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T

### Related Papers
- Esteva et al. (2017): Dermatologist-level classification of skin cancer
- Codella et al. (2019): Skin lesion analysis toward melanoma detection

---

## Support

If you encounter issues:

1. **Check Colab Runtime**: Runtime → View runtime logs
2. **API Status**: https://status.cloud.google.com/
3. **GitHub Issues**: Create an issue in the repository
4. **Course Forum**: Post in your course discussion board

---

**Last Updated**: November 12, 2025  
**Compatible With**: Google Colab (Free/Pro)  
**Python Version**: 3.10+  
**Gemini Model**: gemini-1.5-flash
