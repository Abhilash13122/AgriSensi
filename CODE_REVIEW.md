# AgriSense AI - Comprehensive Code Review

**Date:** June 14, 2026  
**Project:** Agricultural Disease Detection & Predictive Alert System  
**Status:** Production-Ready with Recommendations

---

## Executive Summary

The AgriSense AI application is a well-structured multimodal crop disease diagnostic system with three key features:
1. **Real-time Image Diagnosis** - CNN-based leaf classification
2. **Early Disease Prediction** - LSTM time-series forecasting
3. **Climate-Based Prediction** - XGBoost weather-driven prediction

The codebase demonstrates solid engineering practices but has several areas for improvement in security, error handling, code organization, and performance.

---

## 1. BACKEND CODE REVIEW (`app.py`)

### ✅ Strengths

- **Modular Architecture**: Clear separation of concerns (models, database, RAG engine)
- **Comprehensive Model Loading**: Handles TensorFlow/Keras models, LSTM encoders, XGBoost models gracefully
- **CORS Enabled**: Properly configured for frontend-backend communication
- **Background RAG Initialization**: Non-blocking RAG engine startup
- **Weather Integration**: Both real API and simulation fallback

### ⚠️ Issues Found

#### 1.1 Security Issues

**🔴 CRITICAL - Hardcoded API Keys**
```python
OWM_API_KEY = os.getenv("OWM_API_KEY", "f80c718dad7e26fb52c3acf3e39329e7")
```
**Risk**: API key is exposed in source code
**Fix**: 
- Use `.env` file only (no hardcoded defaults)
- Rotate this API key immediately
- Use environment variable checking

#### 1.2 Error Handling Issues

**🟠 MEDIUM - Insufficient Error Handling in Model Predictions**
```python
@app.post("/predict")
async def predict_api(file: UploadFile = File(...), crop: str = Form(...)):
    if model is None:
        return JSONResponse({"success": False, "error": "Model not loaded properly on server"}, status_code=500)
    
    contents = await file.read()  # No try-catch for large files
```

**Problems**:
- No file size validation (DoS vulnerability)
- No validation of crop parameter (SQL injection risk if used directly)
- Image decode can fail without handling
- `model.predict()` can throw memory errors

**Fix**: Add validation and exception handling:
```python
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
VALID_CROPS = {'Tomato', 'Corn', 'Cotton', 'Sugarcane'}

@app.post("/predict")
async def predict_api(file: UploadFile = File(...), crop: str = Form(...)):
    # Validate crop
    if crop not in VALID_CROPS:
        return JSONResponse({"success": False, "error": "Invalid crop"}, status_code=400)
    
    # Check file size
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        return JSONResponse({"success": False, "error": "File too large"}, status_code=413)
    
    try:
        # Image processing
        img = Image.open(io.BytesIO(contents)).convert('RGB')
        img_resized = img.resize((224, 224))
    except Exception as e:
        return JSONResponse({"success": False, "error": f"Invalid image: {str(e)}"}, status_code=400)
```

#### 1.3 Code Organization

**🟠 MEDIUM - Hard-Coded Paths with Fragile Relative References**
```python
RAG_PDF_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "RAG DATA"))
models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "agrisense_models", "kaggle", "working"))
```

**Issue**: Complex path construction error-prone and platform-dependent

**Fix**: Centralize in config:
```python
# config.py
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "agrisense_models" / "kaggle" / "working"
RAG_DATA_DIR = PROJECT_ROOT / "RAG DATA"
XGB_MODELS_DIR = PROJECT_ROOT / "NEW MODEL WITH 90 ACCURACY"

# Use in app.py
rag_engine = AgriRAGEngine(str(RAG_DATA_DIR))
```

#### 1.4 Performance Issues

**🟠 MEDIUM - Image Resizing Inefficiency**
```python
def estimate_severity(image_cv):
    image_cv = cv2.resize(image_cv, (512, 512))  # ← Large resize for every prediction
```

**Issue**: Resizing to 512x512 every time is computationally expensive

**Recommendation**: Use 256x256 or 384x384 for faster processing while maintaining accuracy

#### 1.5 Logging Issues

**🟠 MEDIUM - No Proper Logging Infrastructure**
```python
print(f"[ERROR] RAG Engine initialization failed: {e}")
print("CNN Model loaded successfully")
```

**Issue**: Using `print()` doesn't persist logs, hard to debug in production

**Fix**: Implement logging:
```python
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.info("CNN Model loaded successfully")
logger.error(f"RAG Engine initialization failed: {e}")
```

---

## 2. DATABASE CODE REVIEW (`database.py`)

### ✅ Strengths

- **Clean SQLite Implementation**: Simple and suitable for this scale
- **Password Hashing**: Uses SHA256 (though not optimal)
- **Email Normalization**: Converts to lowercase for consistency

### ⚠️ Issues Found

#### 2.1 Security Issue - Weak Password Hashing

**🔴 CRITICAL - SHA256 Hash Without Salt**
```python
def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode('utf-8')).hexdigest()
```

**Risk**: 
- Rainbow table attacks possible
- No salt = same password = same hash
- SHA256 not designed for passwords

**Fix**: Use bcrypt or argon2
```python
# Install: pip install bcrypt
import bcrypt

def _hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt(rounds=12)).decode('utf-8')

def verify_password(password: str, stored_hash: str) -> bool:
    return bcrypt.checkpw(password.encode('utf-8'), stored_hash.encode('utf-8'))
```

#### 2.2 SQL Injection Risk

**🟠 MEDIUM - While queries are parameterized (good), email validation missing**
```python
c.execute('SELECT ... FROM users WHERE email = ?', (email.lower(),))
```

**Fix**: Add email validation:
```python
import re

def _validate_email(email: str) -> bool:
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def create_user(email: str, name: str, password: str) -> bool:
    if not _validate_email(email):
        return False
    # ... rest of code
```

#### 2.3 Connection Management

**🟠 MEDIUM - No Connection Pooling**
```python
def verify_user(email: str, password: str) -> Optional[Dict]:
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        # ... query ...
        conn.close()
```

**Issue**: New connection per request is inefficient

**Fix**: Use connection context manager or pool:
```python
from contextlib import contextmanager

@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
    finally:
        conn.close()

def verify_user(email: str, password: str) -> Optional[Dict]:
    try:
        with get_db() as conn:
            c = conn.cursor()
            c.execute('SELECT ... FROM users WHERE email = ?', (email.lower(),))
            row = c.fetchone()
            if row:
                # ... process
```

---

## 3. FRONTEND CODE REVIEW (`index.html` & `script.js`)

### ✅ Strengths

- **Modern HTML5 Structure**: Semantic markup
- **Responsive Design**: Mobile-first approach with proper meta tags
- **Good UX Patterns**: Drag-and-drop, file preview, loading states
- **Tab Navigation**: Clean tabbed interface

### ⚠️ Issues Found

#### 3.1 Security Issues

**🔴 HIGH - No CSRF Protection**
```javascript
form.addEventListener('submit', async (e) => {
    const response = await fetch('/predict', {
        method: 'POST',
        body: formData
    });
```

**Issue**: No CSRF token validation

**Fix**: Add CSRF token:
```html
<!-- In HTML form -->
<input type="hidden" name="csrf_token" id="csrf_token">
```

```python
# In backend
from fastapi import Depends, Form
from uuid import uuid4
import secrets

csrf_tokens = set()

@app.post("/predict")
async def predict_api(
    file: UploadFile = File(...), 
    crop: str = Form(...),
    csrf_token: str = Form(...)
):
    if csrf_token not in csrf_tokens:
        return JSONResponse({"success": False, "error": "Invalid CSRF token"}, status_code=403)
    csrf_tokens.discard(csrf_token)
```

#### 3.2 Code Quality Issues

**🟠 MEDIUM - No Input Validation Before Upload**
```javascript
form.addEventListener('submit', async (e) => {
    e.preventDefault();
    if (!fileInput.files[0]) {
        alert("Please upload an image first.");
        return;
    }
    // Directly sends without validation
```

**Fix**: Add client-side validation:
```javascript
const ALLOWED_TYPES = ['image/jpeg', 'image/png', 'image/jpg'];
const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB

form.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const file = fileInput.files[0];
    if (!file) {
        alert("Please upload an image first.");
        return;
    }
    
    // Validate file type
    if (!ALLOWED_TYPES.includes(file.type)) {
        alert("Only JPEG/PNG images are allowed.");
        return;
    }
    
    // Validate file size
    if (file.size > MAX_FILE_SIZE) {
        alert("File size must be under 10MB.");
        return;
    }
    
    // Validate crop selection
    const crop = document.getElementById('crop-select').value;
    if (!crop) {
        alert("Please select a crop.");
        return;
    }
    
    // Proceed with upload
    await uploadAndPredict();
});
```

#### 3.3 Error Handling

**🟠 MEDIUM - No Error Handling for Network Failures**
```javascript
const response = await fetch('/predict', {
    method: 'POST',
    body: formData
});

const data = await response.json();

if (data.success) {
    // Process results
}
```

**Problems**:
- No try-catch for network errors
- No handling for non-200 responses
- Assumes `data.success` exists

**Fix**:
```javascript
try {
    const response = await fetch('/predict', {
        method: 'POST',
        body: formData
    });
    
    if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
    }
    
    const data = await response.json();
    
    if (!data.success) {
        alert(`Error: ${data.error || 'Prediction failed'}`);
        return;
    }
    
    // Process results
    displayResults(data);
} catch (error) {
    console.error('Prediction failed:', error);
    alert('Network error. Please try again.');
} finally {
    // Reset loading UI
    btnText.textContent = 'Analyze Leaf Image';
    btnLoader.classList.add('hidden');
    submitBtn.disabled = false;
}
```

#### 3.4 Code Organization

**🟠 MEDIUM - Global Variables & State Management**
```javascript
const form = document.getElementById('prediction-form');
const fileInput = document.getElementById('file-upload');
const dropArea = document.getElementById('drop-area');
// ... many more globals
```

**Issue**: Difficult to maintain, potential naming conflicts

**Fix**: Encapsulate in modules or classes:
```javascript
class DiagnosisTab {
    constructor() {
        this.form = document.getElementById('prediction-form');
        this.fileInput = document.getElementById('file-upload');
        this.dropArea = document.getElementById('drop-area');
        this.init();
    }
    
    init() {
        this.form.addEventListener('submit', (e) => this.handleSubmit(e));
        this.fileInput.addEventListener('change', (e) => this.handleFileChange(e));
    }
    
    async handleSubmit(e) {
        // ... logic
    }
}

// Usage
const diagnosis = new DiagnosisTab();
```

#### 3.5 Accessibility Issues

**🟠 MEDIUM - Missing Accessibility Attributes**
```html
<button class="btn-primary" id="submit-btn">
    <span>Analyze Leaf Image</span>
    <div class="loader hidden" id="btn-loader"></div>
</button>
```

**Issues**:
- No `aria-label` for icon buttons
- No `aria-busy` during loading
- Missing `role` attributes where needed

**Fix**:
```html
<button class="btn-primary" id="submit-btn" aria-label="Analyze leaf image">
    <span>Analyze Leaf Image</span>
    <div class="loader hidden" id="btn-loader" aria-hidden="true"></div>
</button>
```

---

## 4. RAG ENGINE CODE REVIEW (`rag_engine.py`)

### ✅ Strengths

- **Comprehensive System Prompt**: Well-structured, farmer-friendly response format
- **Graceful Dependency Handling**: Checks for PyPDF2, sklearn, Gemini
- **Modular Design**: Separate concerns for PDF processing, vectorization, LLM

### ⚠️ Issues Found

#### 4.1 Error Handling

**🟠 MEDIUM - No Error Recovery for LLM API Failures**
- No retry logic for Gemini API timeouts
- No rate limiting handling
- No fallback for API errors

**Fix**: Add retry mechanism:
```python
import time
from functools import wraps

def retry_with_backoff(max_retries=3, backoff_factor=2):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    wait_time = backoff_factor ** attempt
                    logger.warning(f"Attempt {attempt + 1} failed. Retrying in {wait_time}s")
                    time.sleep(wait_time)
        return wrapper
    return decorator

@retry_with_backoff(max_retries=3)
def query_gemini(prompt):
    # API call
```

#### 4.2 Security Issues

**🟠 MEDIUM - No Input Sanitization for User Queries**
```python
def query_rag(user_query):
    # Directly uses user_query in prompt
```

**Risk**: Prompt injection attacks

**Fix**:
```python
import re

def sanitize_query(query: str) -> str:
    # Remove suspicious patterns
    query = re.sub(r'[<>\"\'`]', '', query)
    # Limit length
    query = query[:500]
    return query

def query_rag(user_query: str):
    safe_query = sanitize_query(user_query)
    # Use safe_query
```

---

## 5. MODEL & DATA PIPELINE

### ✅ Strengths

- **Multi-Model Approach**: CNN for classification, LSTM for forecasting, XGBoost for climate
- **Ensemble Methods**: Combining multiple prediction types
- **Severity Estimation**: HSV-based image analysis for infection quantification

### ⚠️ Issues Found

#### 5.1 Model Loading

**🟠 MEDIUM - No Model Versioning**
```python
model_path = os.path.join(os.path.dirname(__file__), "crop_disease_model (1).keras")
```

**Issue**: File naming `(1)` suggests manual versioning, hard to track

**Fix**: Implement semantic versioning:
```
models/
  v1.0.0/
    crop_disease_model.keras
    metadata.json  # Contains accuracy, date, dataset info
  v1.0.1/
    crop_disease_model.keras
    metadata.json

# In app.py
ACTIVE_MODEL_VERSION = "v1.0.1"
model_path = os.path.join(MODELS_DIR, ACTIVE_MODEL_VERSION, "crop_disease_model.keras")
```

#### 5.2 Model Prediction Issues

**🟠 MEDIUM - Hardcoded Confidence Threshold**
```python
if confidence < 0.3:
    disease_name = "Unknown Disease"
```

**Issue**: Threshold should be configurable per model/use-case

**Fix**:
```python
MODEL_CONFIG = {
    "confidence_threshold": 0.3,
    "top_k": 3,
    "apply_temperature": True,
    "temperature": 0.7
}

if confidence < MODEL_CONFIG["confidence_threshold"]:
    disease_name = "Unknown Disease"
```

#### 5.3 Class Name Inconsistencies

**🟠 MEDIUM - Inconsistent Class Naming**
```python
class_names = [
    'Corn_Blight',              # Underscore
    'Cotton_Alternaria Leaf Spot',  # Space
    'Tomato_Bacterial_spot',    # Underscore
    'Tomato__Target_Spot',      # Double underscore
]
```

**Issue**: Inconsistent naming makes parsing and display difficult

**Fix**: Standardize and create mapping:
```python
class_names = [
    'Corn_Blight',
    'Cotton_Alternaria_Leaf_Spot',  # Normalized
    'Tomato_Bacterial_Spot',
    'Tomato_Target_Spot',  # Single underscore
]

DISEASE_DISPLAY_NAMES = {
    'Corn_Blight': 'Corn Blight',
    'Cotton_Alternaria_Leaf_Spot': 'Cotton Alternaria Leaf Spot',
    # ... etc
}

def get_display_name(class_name: str) -> str:
    return DISEASE_DISPLAY_NAMES.get(class_name, class_name.replace('_', ' '))
```

---

## 6. INFRASTRUCTURE & DEPLOYMENT

### 🟠 ISSUES

#### 6.1 No Configuration Management

**Issue**: No `.env` template or documented environment variables

**Fix**: Create `.env.example`:
```bash
# .env.example
GEMINI_API_KEY=your_api_key_here
OWM_API_KEY=your_openweathermap_key_here
DATABASE_PATH=./backend/users.db
LOG_LEVEL=INFO
MAX_FILE_SIZE=10485760
MODEL_VERSION=v1.0.1
```

#### 6.2 No Docker Support

**Issue**: Difficult to deploy consistently

**Recommendation**: Create `Dockerfile` and `docker-compose.yml`

#### 6.3 Missing Tests

**Issue**: No unit or integration tests

**Recommendation**: Add pytest test suite

---

## 7. DEPENDENCIES REVIEW

### Current Dependencies (requirements.txt)

```
fastapi==0.104.1              ✅ OK
tensorflow==2.14.0            ⚠️  Very large, consider TFLite for production
keras==2.14.0                 ✅ OK (now integrated in TF)
opencv-python==4.8.1.78       ✅ OK
google-generativeai==0.3.0    ⚠️  Outdated version (check latest)
```

### Recommendations

```
# Update to latest stable
google-generativeai>=0.5.0

# Add for security
bcrypt>=4.0.0
python-jose>=3.3.0

# Add for logging
python-json-logger>=2.0.0

# Add for testing
pytest>=7.0.0
pytest-asyncio>=0.20.0
httpx>=0.24.0
```

---

## 8. QUICK WINS (Priority Order)

### 🔴 CRITICAL (Do First)
1. Remove hardcoded API key - rotate immediately
2. Change SHA256 password hashing to bcrypt
3. Add CSRF protection to frontend
4. Add input validation and file size limits

### 🟠 HIGH (Do This Week)
5. Implement proper logging
6. Add error handling for network failures
7. Centralize configuration
8. Add comprehensive error handling in `/predict` endpoint

### 🟡 MEDIUM (Do This Sprint)
9. Refactor frontend with modules/classes
10. Implement model versioning
11. Add unit tests
12. Document API endpoints

### 🟢 LOW (Future)
13. Add Docker support
14. Implement rate limiting
15. Add request caching
16. Performance optimization

---

## 9. TESTING CHECKLIST

### Unit Tests Needed
- [ ] Password hashing and verification
- [ ] User creation and authentication
- [ ] Image validation
- [ ] Model inference with different crops
- [ ] RAG query handling

### Integration Tests Needed
- [ ] Full prediction pipeline (image → diagnosis)
- [ ] Weather API fallback
- [ ] User workflow (signup → prediction → history)

### Security Tests Needed
- [ ] CSRF token validation
- [ ] File upload validation
- [ ] SQL injection attempts
- [ ] XSS attempts

---

## 10. PERFORMANCE OPTIMIZATION OPPORTUNITIES

1. **Model Optimization**
   - Quantize models to int8 for faster inference
   - Consider model pruning
   - Use TensorFlow Lite for mobile

2. **Image Processing**
   - Reduce resize dimensions from 512x512 to 384x384
   - Use async image processing

3. **Caching**
   - Cache model predictions for identical images
   - Cache RAG responses for common queries

4. **Database**
   - Add indexes on email column
   - Consider PostgreSQL for production

---

## CONCLUSION

**Overall Assessment**: **7.5/10 - Good Foundation, Production-Ready with Improvements**

The application demonstrates solid engineering with proper separation of concerns, error handling, and user experience. However, critical security issues (exposed API key, weak password hashing, no CSRF) need immediate attention before production deployment.

**Recommended Next Steps**:
1. Address critical security issues immediately
2. Implement comprehensive error handling
3. Add unit and integration tests
4. Set up CI/CD pipeline
5. Document API and deployment procedures

**Estimated Effort to Production-Ready**: 2-3 weeks (1 week critical fixes, 1-2 weeks improvements)

