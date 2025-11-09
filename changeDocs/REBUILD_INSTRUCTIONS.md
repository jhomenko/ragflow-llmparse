# Docker Clean Rebuild Instructions

## Problem
Docker is using cached layers, so code changes aren't being applied to the container.

## Solution: Force Clean Rebuild

### Step 1: Stop and Remove Everything
```bash
# Stop all containers
docker-compose down

# Remove the RAGFlow container specifically
docker rm -f ragflow-server 2>/dev/null || true

# Remove the RAGFlow image
docker rmi ragflow/ragflow:latest 2>/dev/null || true

# Optional: Clean all unused Docker resources
docker system prune -f
```

### Step 2: Rebuild Without Cache
```bash
# Build with --no-cache to force rebuild of all layers
docker-compose build --no-cache

# Or if using specific compose file:
docker-compose -f docker/docker-compose-CN-oc9.yml build --no-cache
```

### Step 3: Start Services
```bash
# Start the stack
docker-compose up -d

# Watch logs
docker-compose logs -f ragflow
```

### Step 4: Verify Changes Are Applied
```bash
# Enter the container
docker exec -it ragflow-server bash

# Check if code changes are present
# Bug #1 fix:
grep -A2 "def describe_with_prompt" /ragflow/api/db/services/llm_service.py | tail -1
# Should show: return txt, used_tokens

# Bug #2 fix:
grep -A5 "def vision_llm_prompt" /ragflow/rag/llm/cv_model.py | grep "system"
# Should show system message

# Check test file exists
ls -la /ragflow/test_vlm_fix.py
```

## Alternative: Build from Dockerfile Directly

If compose caching persists:

```bash
# Build image directly
docker build --no-cache -t ragflow/ragflow:latest -f Dockerfile .

# Then start with compose
docker-compose up -d
```

## Running the Test Script

### Inside Docker Container
```bash
# Enter container
docker exec -it ragflow-server bash

# Run test (PDF must be accessible in container)
cd /ragflow
python test_vlm_fix.py /path/to/pdf/in/container.pdf
```

### From Host (Preferred)
```bash
# Copy PDF into container
docker cp your-test.pdf ragflow-server:/tmp/test.pdf

# Run test
docker exec ragflow-server python /ragflow/test_vlm_fix.py /tmp/test.pdf
```

## Verification Checklist

After rebuild, verify:
- [ ] `api/db/services/llm_service.py` line 165 returns tuple
- [ ] `rag/llm/cv_model.py` includes system message
- [ ] `rag/llm/cv_model.py` includes max_tokens=4096
- [ ] `test_vlm_fix.py` exists in container