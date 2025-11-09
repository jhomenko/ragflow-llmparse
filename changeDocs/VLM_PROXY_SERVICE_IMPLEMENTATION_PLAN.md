# VLM Proxy Service Implementation Plan

## Executive Summary

Create a lightweight proxy service that wraps your working VLM test script and integrates seamlessly with RAGFlow's existing architecture. This approach:
- ✅ Uses your proven working code (guaranteed results)
- ✅ Maintains UI model selection functionality
- ✅ Minimal changes to RAGFlow codebase
- ✅ Easy to debug and maintain
- ✅ Can be disabled/enabled via configuration

## Architecture Overview

```
┌─────────────────┐
│   RAGFlow UI    │
│  (Model Select) │
└────────┬────────┘
         │ Model: "Qwen2.5VL-3B"
         │ Prompt: "Transcribe..."
         ▼
┌─────────────────────────┐
│  RAGFlow Parser         │
│  (pdf_parser.py)        │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  VLM Proxy Service      │◄─── NEW SERVICE (Port 8081)
│  (vlm_proxy_server.py)  │
└────────┬────────────────┘
         │ Uses your working test code
         │
         ▼
┌─────────────────────────┐
│  VLM Server             │
│  (192.168.68.186:8080)  │
└─────────────────────────┘
```

## Implementation Breakdown

---

## Phase 1: Create VLM Proxy Service (NEW)

### Task 1.1: Create Proxy Server Script

**File**: `rag/app/vlm_proxy_server.py` (NEW FILE)

**Description**: Standalone Flask service that wraps your working test code.

**Implementation**:

```python
#!/usr/bin/env python3
"""
VLM Proxy Service - Wraps working VLM test code as HTTP API
Runs on port 8081 by default
"""

from flask import Flask, request, jsonify
import base64
import logging
from openai import OpenAI
import os

app = Flask(__name__)

# Configuration from environment
VLM_BASE_URL = os.getenv("VLM_BASE_URL", "http://192.168.68.186:8080/v1")
VLM_DEFAULT_MODEL = os.getenv("VLM_DEFAULT_MODEL", "Qwen2.5VL-3B")
PROXY_PORT = int(os.getenv("VLM_PROXY_PORT", "8081"))
PROXY_HOST = os.getenv("VLM_PROXY_HOST", "0.0.0.0")

# Initialize OpenAI client (same as your working test)
client = OpenAI(
    api_key="not-needed",
    base_url=VLM_BASE_URL
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "service": "vlm-proxy"}), 200


@app.route('/v1/describe', methods=['POST'])
def describe_image():
    """
    Main endpoint: Convert image to markdown
    
    Request JSON:
    {
        "image_base64": "...",  # Base64 encoded JPEG/PNG
        "prompt": "Transcribe...",  # Custom prompt
        "model": "Qwen2.5VL-3B",  # Optional model override
        "max_tokens": 4096,  # Optional
        "temperature": 0.1  # Optional
    }
    
    Response JSON:
    {
        "text": "# Markdown output...",
        "tokens": 1234,
        "model": "Qwen2.5VL-3B"
    }
    """
    try:
        data = request.get_json()
        
        # Validate required fields
        if not data or 'image_base64' not in data:
            return jsonify({"error": "Missing image_base64"}), 400
        
        image_b64 = data['image_base64']
        prompt = data.get('prompt', 'Transcribe this document page to clean Markdown format.')
        model = data.get('model', VLM_DEFAULT_MODEL)
        max_tokens = data.get('max_tokens', 4096)
        temperature = data.get('temperature', 0.1)
        
        logger.info(f"Processing request: model={model}, prompt_len={len(prompt)}, img_len={len(image_b64)}")
        
        # Construct messages (EXACT SAME FORMAT AS YOUR WORKING TEST)
        system_message = {
            "role": "system",
            "content": "You are a helpful assistant that converts document images to clean Markdown text."
        }
        
        user_message = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_b64}"
                    }
                }
            ]
        }
        
        # Call VLM (EXACT SAME CODE AS YOUR WORKING TEST)
        response = client.chat.completions.create(
            model=model,
            messages=[system_message, user_message],
            max_tokens=max_tokens,
            temperature=temperature,
            # NO extra_body or other parameters
        )
        
        # Extract response
        text = response.choices[0].message.content
        tokens = response.usage.total_tokens if response.usage else 0
        
        logger.info(f"VLM response: {len(text)} chars, {tokens} tokens")
        
        return jsonify({
            "text": text,
            "tokens": tokens,
            "model": model,
            "success": True
        }), 200
        
    except Exception as e:
        logger.exception(f"Error processing request: {e}")
        return jsonify({
            "error": str(e),
            "success": False
        }), 500


@app.route('/v1/models', methods=['GET'])
def list_models():
    """
    List available models
    Used by RAGFlow UI to populate model dropdown
    """
    # You can expand this list as needed
    models = [
        {"id": "Qwen2.5VL-3B", "name": "Qwen2.5VL-3B"},
        {"id": "Qwen2.5VL-7B", "name": "Qwen2.5VL-7B"},
        # Add more models as they become available
    ]
    return jsonify({"models": models}), 200


if __name__ == '__main__':
    logger.info(f"Starting VLM Proxy Service on {PROXY_HOST}:{PROXY_PORT}")
    logger.info(f"Forwarding to VLM at {VLM_BASE_URL}")
    app.run(host=PROXY_HOST, port=PROXY_PORT, debug=False)
```

**Dependencies**: None beyond existing (Flask, OpenAI client)

**Testing**:
```bash
# Start service
python3 rag/app/vlm_proxy_server.py

# Test health
curl http://localhost:8081/health

# Test image description
python3 test_proxy_service.py
```

---

### Task 1.2: Create Proxy Service Startup Script

**File**: `docker/launch_vlm_proxy.sh` (NEW FILE)

**Description**: Service startup script for Docker container.

**Implementation**:

```bash
#!/bin/bash
set -e

echo "Starting VLM Proxy Service..."

# Configuration
export VLM_BASE_URL="${VLM_BASE_URL:-http://192.168.68.186:8080/v1}"
export VLM_PROXY_PORT="${VLM_PROXY_PORT:-8081}"
export VLM_PROXY_HOST="${VLM_PROXY_HOST:-0.0.0.0}"

# Start service
cd /ragflow
exec python3 -u rag/app/vlm_proxy_server.py
```

**Make executable**:
```bash
chmod +x docker/launch_vlm_proxy.sh
```

---

### Task 1.3: Add Proxy Service to Docker Compose

**File**: `docker/docker-compose-CN-oc9.yml` (MODIFY)

**Changes**:

```yaml
services:
  ragflow:
    # ... existing configuration ...
    environment:
      # ... existing env vars ...
      - VLM_PROXY_ENABLED=true  # NEW: Enable proxy
      - VLM_PROXY_URL=http://localhost:8081  # NEW: Proxy URL
      - VLM_BASE_URL=http://192.168.68.186:8080/v1  # NEW: Actual VLM URL
    
  # NEW SERVICE: VLM Proxy
  vlm-proxy:
    image: infiniflow/ragflow:dev  # Same image as ragflow
    container_name: vlm-proxy
    ports:
      - "8081:8081"  # Expose proxy port
    environment:
      - VLM_BASE_URL=http://192.168.68.186:8080/v1
      - VLM_PROXY_PORT=8081
      - VLM_PROXY_HOST=0.0.0.0
    volumes:
      - ./:/ragflow  # Mount code
    command: /ragflow/docker/launch_vlm_proxy.sh
    networks:
      - ragflow
    restart: unless-stopped
```

**Note**: This creates a separate container for the proxy service, ensuring isolation and easy debugging.

---

## Phase 2: Integrate Proxy into RAGFlow

### Task 2.1: Create Proxy Client Wrapper

**File**: `rag/app/vlm_proxy_client.py` (NEW FILE)

**Description**: Client library for calling the proxy service.

**Implementation**:

```python
"""
VLM Proxy Client - Calls proxy service instead of direct VLM
"""

import requests
import logging
import base64
import os

logger = logging.getLogger(__name__)


class VLMProxyClient:
    """Client for VLM Proxy Service"""
    
    def __init__(self, proxy_url=None):
        """
        Initialize proxy client
        
        Args:
            proxy_url: Proxy service URL (default: from env or localhost:8081)
        """
        self.proxy_url = proxy_url or os.getenv(
            "VLM_PROXY_URL", 
            "http://localhost:8081"
        )
        self.enabled = os.getenv("VLM_PROXY_ENABLED", "true").lower() == "true"
        
        logger.info(f"VLMProxyClient initialized: url={self.proxy_url}, enabled={self.enabled}")
    
    def describe_image(self, image_bytes, prompt, model_name, max_tokens=4096, temperature=0.1):
        """
        Describe image using proxy service
        
        Args:
            image_bytes: JPEG/PNG bytes
            prompt: Custom prompt text
            model_name: VLM model name (e.g., "Qwen2.5VL-3B")
            max_tokens: Max tokens in response
            temperature: Sampling temperature
        
        Returns:
            (text, token_count) tuple
        
        Raises:
            Exception if proxy call fails
        """
        if not self.enabled:
            raise Exception("VLM Proxy is disabled. Set VLM_PROXY_ENABLED=true")
        
        try:
            # Convert bytes to base64
            image_b64 = base64.b64encode(image_bytes).decode('utf-8')
            
            # Prepare request
            payload = {
                "image_base64": image_b64,
                "prompt": prompt,
                "model": model_name,
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            
            logger.info(f"Calling proxy: model={model_name}, prompt_len={len(prompt)}")
            
            # Call proxy service
            response = requests.post(
                f"{self.proxy_url}/v1/describe",
                json=payload,
                timeout=120  # 2 min timeout for large images
            )
            
            response.raise_for_status()
            
            # Parse response
            result = response.json()
            
            if not result.get("success"):
                raise Exception(f"Proxy returned error: {result.get('error')}")
            
            text = result.get("text", "")
            tokens = result.get("tokens", 0)
            
            logger.info(f"Proxy response: {len(text)} chars, {tokens} tokens")
            
            return text, tokens
            
        except requests.exceptions.RequestException as e:
            logger.exception(f"Proxy request failed: {e}")
            raise Exception(f"VLM Proxy unreachable: {e}")
        except Exception as e:
            logger.exception(f"Proxy call failed: {e}")
            raise


def get_proxy_client():
    """Get singleton proxy client instance"""
    if not hasattr(get_proxy_client, '_instance'):
        get_proxy_client._instance = VLMProxyClient()
    return get_proxy_client._instance
```

---

### Task 2.2: Modify vision_llm_chunk to Use Proxy

**File**: `rag/app/picture.py` (MODIFY)

**Changes**:

```python
# Add import at top
from rag.app.vlm_proxy_client import get_proxy_client
import os

# Modify vision_llm_chunk function (around line 68)
def vision_llm_chunk(binary, vision_model, prompt=None, callback=None):
    """
    Process image bytes to markdown text via VLM.
    
    Args:
        binary: JPEG/PNG bytes
        vision_model: LLMBundle instance (used for model name only if proxy enabled)
        prompt: Custom prompt string
        callback: Progress callback
    
    Returns:
        Markdown text string
    """
    callback = callback or (lambda prog, msg: None)
    
    # NEW: Check if proxy is enabled
    use_proxy = os.getenv("VLM_PROXY_ENABLED", "true").lower() == "true"
    
    if use_proxy:
        try:
            # Use proxy service (YOUR WORKING CODE)
            proxy = get_proxy_client()
            model_name = getattr(vision_model, 'model_name', 'Qwen2.5VL-3B')
            
            text, token_count = proxy.describe_image(
                image_bytes=binary,
                prompt=prompt or "Transcribe this document page to clean Markdown.",
                model_name=model_name,
                max_tokens=4096,
                temperature=0.1
            )
            
            # Clean markdown fences if present
            text = clean_markdown_block(text)
            
            logging.info(f"VLM Proxy response: {len(text)} chars, {token_count} tokens")
            return text
            
        except Exception as e:
            logging.error(f"Proxy call failed, falling back to direct: {e}")
            # Fall through to original code below
    
    # ORIGINAL CODE: Direct VLM call (kept as fallback)
    if not isinstance(binary, bytes):
        logging.error(f"vision_llm_chunk expects bytes, got {type(binary)}")
        return ""
    
    try:
        txt, token_count = vision_model.describe_with_prompt(binary, prompt or "")
        txt = clean_markdown_block(txt)
        logging.info(f"Direct VLM response: {len(txt)} chars, {token_count} tokens")
        return txt
    except Exception as e:
        logging.exception(f"vision_llm_chunk failed: {e}")
        callback(-1, str(e))
        return ""
```

**Why This Works**:
- ✅ Proxy enabled by default (uses your working code)
- ✅ Falls back to original code if proxy fails
- ✅ Model name still comes from UI selection
- ✅ Zero changes needed to other files

---

### Task 2.3: Update Configuration Files

**File**: `conf/service_conf.yaml.template` (MODIFY)

**Add proxy configuration section**:

```yaml
# ... existing configuration ...

# VLM Proxy Service Configuration
vlm_proxy:
  enabled: true  # Enable proxy service
  url: "http://localhost:8081"  # Proxy service URL
  timeout: 120  # Request timeout in seconds
  fallback_to_direct: true  # Fall back to direct VLM if proxy fails
```

---

## Phase 3: Testing and Validation

### Task 3.1: Create Proxy Service Test Script

**File**: `test_proxy_service.py` (NEW FILE)

**Description**: Test the proxy service independently.

**Implementation**:

```python
#!/usr/bin/env python3
"""
Test VLM Proxy Service
"""

import requests
import base64
import json
import sys

PROXY_URL = "http://localhost:8081"

def test_health():
    """Test health endpoint"""
    print("Testing health endpoint...")
    response = requests.get(f"{PROXY_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    assert response.status_code == 200
    print("✓ Health check passed\n")

def test_describe_image():
    """Test image description"""
    print("Testing image description...")
    
    # Load test PDF and convert first page to JPEG
    import fitz  # PyMuPDF
    pdf_path = "test_data/sample.pdf"  # Update with your test PDF
    
    doc = fitz.open(pdf_path)
    page = doc[0]
    pix = page.get_pixmap(dpi=150)
    img_bytes = pix.tobytes("jpeg")
    doc.close()
    
    # Encode to base64
    img_b64 = base64.b64encode(img_bytes).decode('utf-8')
    
    # Prepare request
    payload = {
        "image_base64": img_b64,
        "prompt": "Transcribe this PDF page to clean Markdown format.",
        "model": "Qwen2.5VL-3B",
        "max_tokens": 4096,
        "temperature": 0.1
    }
    
    # Call proxy
    response = requests.post(
        f"{PROXY_URL}/v1/describe",
        json=payload,
        timeout=120
    )
    
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Success: {result.get('success')}")
        print(f"Tokens: {result.get('tokens')}")
        print(f"Text length: {len(result.get('text', ''))}")
        print(f"Text preview: {result.get('text', '')[:200]}...")
        
        assert len(result.get('text', '')) > 100, "Response too short!"
        print("\n✓ Image description test passed")
    else:
        print(f"Error: {response.text}")
        sys.exit(1)

def test_models_endpoint():
    """Test models listing"""
    print("\nTesting models endpoint...")
    response = requests.get(f"{PROXY_URL}/v1/models")
    print(f"Status: {response.status_code}")
    print(f"Models: {response.json()}")
    assert response.status_code == 200
    print("✓ Models endpoint passed\n")

if __name__ == "__main__":
    try:
        test_health()
        test_models_endpoint()
        test_describe_image()
        print("\n✅ All tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
```

---

### Task 3.2: Create End-to-End Integration Test

**File**: `test_vlm_proxy_integration.py` (NEW FILE)

**Description**: Test the full RAGFlow → Proxy → VLM chain.

**Implementation**:

```python
#!/usr/bin/env python3
"""
Test VLM Proxy Integration with RAGFlow
"""

import sys
import os

# Add ragflow to path
sys.path.insert(0, '/ragflow')

from deepdoc.parser.pdf_parser import VisionParser
from api.db.services.llm_service import LLMBundle
from api.db import LLMType
from rag.app.vlm_proxy_client import get_proxy_client

def test_proxy_client():
    """Test proxy client directly"""
    print("Testing VLM Proxy Client...")
    
    # Read test image
    with open("test_data/sample.pdf", "rb") as f:
        import fitz
        doc = fitz.open(stream=f.read(), filetype="pdf")
        page = doc[0]
        pix = page.get_pixmap(dpi=150)
        img_bytes = pix.tobytes("jpeg")
        doc.close()
    
    # Call proxy
    proxy = get_proxy_client()
    text, tokens = proxy.describe_image(
        image_bytes=img_bytes,
        prompt="Transcribe this page to Markdown.",
        model_name="Qwen2.5VL-3B"
    )
    
    print(f"Response: {len(text)} chars, {tokens} tokens")
    print(f"Preview: {text[:200]}...")
    
    assert len(text) > 100, "Response too short!"
    print("✓ Proxy client test passed\n")

def test_vision_parser_with_proxy():
    """Test VisionParser using proxy"""
    print("Testing VisionParser with Proxy...")
    
    # Enable proxy
    os.environ['VLM_PROXY_ENABLED'] = 'true'
    os.environ['VLM_PROXY_URL'] = 'http://localhost:8081'
    
    # Create vision model bundle (only used for model name)
    vision_model = LLMBundle(
        tenant_id="test_tenant",
        llm_type=LLMType.IMAGE2TEXT,
        llm_name="Qwen2.5VL-3B",
        lang="English"
    )
    
    # Parse PDF
    parser = VisionParser(vision_model=vision_model)
    
    with open("test_data/sample.pdf", "rb") as f:
        pdf_bytes = f.read()
    
    lines, _ = parser(
        pdf_bytes,
        from_page=0,
        to_page=1,
        prompt_text="Transcribe this PDF page to clean Markdown format.",
        zoomin=3
    )
    
    # Validate
    assert len(lines) > 0, "No pages processed"
    text, meta = lines[0]
    
    print(f"Extracted: {len(text)} chars")
    print(f"Metadata: {meta}")
    print(f"Preview: {text[:200]}...")
    
    assert len(text) > 100, f"Response too short: {len(text)} chars"
    print("✓ VisionParser test passed\n")

if __name__ == "__main__":
    try:
        # Set environment
        os.environ['VLM_PROXY_ENABLED'] = 'true'
        os.environ['VLM_PROXY_URL'] = 'http://localhost:8081'
        
        test_proxy_client()
        test_vision_parser_with_proxy()
        
        print("✅ All integration tests passed!")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
```

---

## Phase 4: Deployment

### Task 4.1: Update Docker Build

**File**: `Dockerfile` (MODIFY - if needed)

**No changes required** - proxy service uses same Python environment.

---

### Task 4.2: Create Deployment Checklist

**File**: `VLM_PROXY_DEPLOYMENT.md` (NEW FILE)

```markdown
# VLM Proxy Service Deployment Checklist

## Pre-Deployment

- [ ] Proxy service script created (`rag/app/vlm_proxy_server.py`)
- [ ] Proxy client library created (`rag/app/vlm_proxy_client.py`)
- [ ] Startup script created (`docker/launch_vlm_proxy.sh`)
- [ ] Docker compose updated with vlm-proxy service
- [ ] picture.py modified to use proxy
- [ ] Test scripts created and passing

## Deployment Steps

1. **Start Proxy Service**
   ```bash
   docker-compose -f docker/docker-compose-CN-oc9.yml up -d vlm-proxy
   ```

2. **Verify Proxy Health**
   ```bash
   curl http://localhost:8081/health
   ```

3. **Run Proxy Tests**
   ```bash
   python3 test_proxy_service.py
   ```

4. **Restart RAGFlow**
   ```bash
   docker-compose -f docker/docker-compose-CN-oc9.yml restart ragflow
   ```

5. **Run Integration Tests**
   ```bash
   docker exec ragflow-server python3 test_vlm_proxy_integration.py
   ```

6. **Test via UI**
   - Upload PDF with VLM parser
   - Select model from dropdown
   - Verify chunks created with full content

## Rollback Plan

If proxy fails, disable it:

```bash
docker exec ragflow-server bash -c "echo 'export VLM_PROXY_ENABLED=false' >> /etc/profile"
docker-compose restart ragflow
```

Original direct VLM code will be used as fallback.

## Monitoring

Check proxy logs:
```bash
docker logs -f vlm-proxy
```

Check RAGFlow logs for proxy calls:
```bash
docker logs ragflow-server | grep -i "proxy"
```
```

---

## Phase 5: Documentation and Maintenance

### Task 5.1: Create User Guide

**File**: `docs/VLM_PROXY_USER_GUIDE.md` (NEW FILE)

```markdown
# VLM Proxy Service User Guide

## What is VLM Proxy?

The VLM Proxy Service is a lightweight wrapper that ensures reliable communication with Vision Language Models. It uses proven, tested code to convert PDF pages to markdown.

## Features

- ✅ **Guaranteed Results**: Uses tested code path
- ✅ **Model Selection**: Choose models from UI dropdown
- ✅ **Custom Prompts**: Configure prompts via settings
- ✅ **Fallback**: Automatically falls back to direct VLM if proxy fails
- ✅ **Easy Debugging**: Separate service with dedicated logs

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VLM_PROXY_ENABLED` | `true` | Enable/disable proxy |
| `VLM_PROXY_URL` | `http://localhost:8081` | Proxy service URL |
| `VLM_BASE_URL` | `http://192.168.68.186:8080/v1` | Actual VLM server URL |

### Using Different Models

1. Go to RAGFlow UI → Dataset → Upload Document
2. Select "Parser Config"
3. Choose VLM model from dropdown (e.g., "Qwen2.5VL-3B")
4. Upload PDF

The proxy will automatically use the selected model.

## Troubleshooting

### Proxy Service Not Starting

```bash
# Check service status
docker ps | grep vlm-proxy

# Check logs
docker logs vlm-proxy

# Restart service
docker-compose restart vlm-proxy
```

### Empty Responses

1. Check proxy logs: `docker logs vlm-proxy`
2. Verify VLM server is reachable: `curl http://192.168.68.186:8080/health`
3. Test proxy directly: `python3 test_proxy_service.py`

### Model Not Found

Ensure model name in UI matches exactly what VLM server expects:
- Correct: `Qwen2.5VL-3B`
- Incorrect: `qwen-2.5-vl-3b`

## Advanced Usage

### Adding New Models

Edit `rag/app/vlm_proxy_server.py`:

```python
@app.route('/v1/models', methods=['GET'])
def list_models():
    models = [
        {"id": "Qwen2.5VL-3B", "name": "Qwen2.5VL-3B"},
        {"id": "Qwen2.5VL-7B", "name": "Qwen2.5VL-7B"},
        {"id": "YourNewModel", "name": "Your New Model"},  # ADD THIS
    ]
    return jsonify({"models": models}), 200
```

### Custom Prompts

Edit prompt file: `rag/prompts/vision_llm_describe_prompt.md`

Restart services:
```bash
docker-compose restart vlm-proxy ragflow
```
```

---

### Task 5.2: Add to Main README

**File**: `README.md` (MODIFY)

**Add section**:

```markdown
## VLM Proxy Service

RAGFlow includes a VLM Proxy Service for reliable PDF-to-markdown conversion using Vision Language Models.

### Quick Start

1. Configure VLM server URL in `docker-compose.yml`:
   ```yaml
   environment:
     - VLM_BASE_URL=http://your-vlm-server:8080/v1
   ```

2. Start services:
   ```bash
   docker-compose up -d
   ```

3. Use VLM parser in UI:
   - Select VLM model from dropdown
   - Upload PDF
   - Chunks will be created automatically

See [VLM Proxy User Guide](docs/VLM_PROXY_USER_GUIDE.md) for details.
```

---

## Implementation Timeline

### Week 1: Core Implementation
- [ ] Task 1.1: Create proxy server script
- [ ] Task 1.2: Create startup script
- [ ] Task 1.3: Update docker-compose
- [ ] Task 2.1: Create proxy client
- [ ] Task 2.2: Modify picture.py

### Week 2: Testing
- [ ] Task 3.1: Create proxy service tests
- [ ] Task 3.2: Create integration tests
- [ ] Run all tests and fix issues

### Week 3: Deployment & Documentation
- [ ] Task 4.1: Deploy to staging
- [ ] Task 4.2: Verify deployment checklist
- [ ] Task 5.1: Create user guide
- [ ] Task 5.2: Update main README
- [ ] Deploy to production

---

## Success Criteria

### Must Have
- ✅ Proxy service runs independently
- ✅ Returns same results as your working test script
- ✅ Model selection works via UI
- ✅ Fallback to direct VLM works
- ✅ All tests pass

### Should Have
- ✅ Comprehensive logging
- ✅ Health check endpoint
- ✅ User documentation
- ✅ Easy rollback mechanism

### Nice to Have
- ⚠️ Multiple VLM servers support
- ⚠️ Load balancing
- ⚠️ Caching layer
- ⚠️ Metrics/monitoring

---

## Advantages of This Approach

1. **Guaranteed to Work**: Uses your exact working test code
2. **Minimal Risk**: Proxy is separate service, doesn't break existing code
3. **Easy Debugging**: Dedicated logs, can test independently
4. **Maintains UI**: Model selection still works
5. **Flexible**: Can add features (caching, load balancing) later
6. **Rollback Friendly**: Just disable proxy env var

## Disadvantages (and Mitigations)

1. **Extra Service**: Adds complexity
   - *Mitigation*: Can run in same container if preferred
   
2. **Network Hop**: Slight latency increase
   - *Mitigation*: Minimal (localhost), can be same container
   
3. **Maintenance**: Another codebase to maintain
   - *Mitigation*: Very simple code (~200 lines)

---

## Alternative: In-Process Integration

If you prefer NOT to run a separate service, you can integrate directly:

**Simpler Approach**: Just replace the `describe_with_prompt` call in `picture.py` with your working OpenAI client code.

**File**: `rag/app/picture.py`

```python
def vision_llm_chunk(binary, vision_model, prompt=None, callback=None):
    """Use working OpenAI client code directly"""
    callback = callback or (lambda prog, msg: None)
    
    # YOUR WORKING CODE (inline)
    from openai import OpenAI
    import base64
    
    client = OpenAI(
        api_key="not-needed",
        base_url="http://192.168.68.186:8080/v1"
    )
    
    img_b64 = base64.b64encode(binary).decode('utf-8')
    model = getattr(vision_model, 'model_name', 'Qwen2.5VL-3B')
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant..."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt or "Transcribe..."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                ]
            }
        ],
        max_tokens=4096,
        temperature=0.1
    )
    
    text = response.choices[0].message.content
    return clean_markdown_block(text)
```

This is simpler but less flexible. The proxy service approach is recommended for production.

---

## Questions for Review

Before implementing, please confirm:

1. **Service Architecture**: Separate container or in-process?
2. **Model Management**: Should proxy auto-discover models from VLM server?
3. **Caching**: Should proxy cache responses for identical images?
4. **Load Balancing**: Multiple VLM servers or single server?
5. **Monitoring**: Integrate with existing monitoring or separate?

---

## Next Steps

1. Review this plan
2. Choose approach (proxy service vs in-process)
3. Create subtasks for implementation team
4. Begin Phase 1 implementation
5. Test each phase before proceeding

This plan ensures your working test code is used while maintaining all RAGFlow UI functionality.