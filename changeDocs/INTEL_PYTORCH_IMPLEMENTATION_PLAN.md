# Intel PyTorch Integration Implementation Plan

## Overview
Integrate Intel Extension for PyTorch (IPEX) into the RAGFlow Docker build process, replacing the default NVIDIA CUDA PyTorch installation with Intel XPU-optimized versions.

## Problem Statement
The current Dockerfile uses UV package manager with a virtual environment that doesn't recognize the Intel PyTorch packages pre-installed in the `intel/intel-extension-for-pytorch:2.8.10-xpu` base image. UV installs the default NVIDIA CUDA versions of PyTorch instead.

## Root Causes Identified
1. **UV Lock File Issue**: Current `uv.lock` references standard PyPI PyTorch packages (CUDA versions)
2. **Python Version Mismatch**: Dockerfile specifies `--python 3.10` but Intel base image likely uses Python 3.11
3. **Missing Index URLs**: Intel-specific package indexes not configured in `pyproject.toml`
4. **Virtual Environment Isolation**: UV's venv doesn't see system-installed Intel packages

## Solution Strategy
Use **system Python installation** approach (Option A) as it's simpler and more reliable for Docker containers:
- Install directly to system Python (no venv isolation needed in containers)
- Ensures UV uses Intel packages already in base image
- Cleaner dependency resolution
- Fewer moving parts = higher success probability

---

## Detailed Implementation Steps

### Step 1: Verify Python Version in Intel Base Image
**Objective**: Detect the actual Python version to ensure alignment

**Actions**:
1. Add Python version detection early in Dockerfile
2. Store detected version in environment variable
3. Use this version consistently throughout the build

**Code Changes** (Dockerfile, after base stage setup):
```dockerfile
# Detect Python version from base image
RUN PYTHON_VERSION=$(python3 --version | grep -oP '3\.\d+') && \
    echo "Detected Python version: $PYTHON_VERSION" && \
    echo "export DETECTED_PYTHON_VERSION=$PYTHON_VERSION" >> /etc/environment
```

**Validation**: Check build logs show detected Python version

---

### Step 2: Update pyproject.toml
**Objective**: Configure Intel PyTorch package sources and add Intel-specific dependencies

**Actions**:
1. Add Intel PyTorch index URLs to `[[tool.uv.index]]` section
2. Add `intel-extension-for-pytorch` and `oneccl_bind_pt` to dependencies
3. Ensure `torch`, `torchvision`, `torchaudio` versions match Intel requirements (2.8.0/0.23.0/2.8.0)

**Code Changes** (pyproject.toml):

```toml
# Add after existing [[tool.uv.index]] section around line 165
[[tool.uv.index]]
name = "pytorch-xpu"
url = "https://download.pytorch.org/whl/xpu"
explicit = true

[[tool.uv.index]]
name = "intel-extension"
url = "https://pytorch-extension.intel.com/release-whl/stable/xpu/us/"
explicit = true
```

**Update dependencies section** (around line 148-150):
```toml
[project.optional-dependencies]
full = [
    "bcembedding==0.1.5",
    "fastembed>=0.3.6,<0.4.0; sys_platform == 'darwin' or platform_machine != 'x86_64'",
    "fastembed-gpu>=0.3.6,<0.4.0; sys_platform != 'darwin' and platform_machine == 'x86_64'",
    "flagembedding==1.2.10",
    "torch==2.8.0",
    "torchvision==0.23.0", 
    "torchaudio==2.8.0",
    "intel-extension-for-pytorch==2.8.10+xpu",
    "oneccl-bind-pt==2.8.0+xpu",
    "transformers>=4.35.0,<5.0.0",
]
```

**Important Notes**:
- Pin exact versions to match Intel base image
- Use `+xpu` suffix for Intel packages
- Keep transformers as-is (works with Intel PyTorch)

**Validation**: Run `uv lock` locally to verify it resolves correctly

---

### Step 3: Generate New uv.lock File
**Objective**: Create lock file with Intel PyTorch dependencies

**Actions** (Local machine, before Docker build):
```bash
# Clean existing lock
rm uv.lock

# Generate new lock with Intel indexes
# Note: This may show warnings about unresolved packages - that's OK for Intel-specific ones
uv lock --no-cache

# Verify the lock file was created
ls -lh uv.lock
```

**Expected Outcome**:
- New `uv.lock` file generated
- File size should be similar to original
- Should NOT contain NVIDIA CUDA packages

**Troubleshooting**:
If `uv lock` fails with resolution errors:
1. Temporarily remove Intel-specific packages from dependencies
2. Run `uv lock` to get base resolution
3. Intel packages will be installed directly from base image

**Validation**: Check `uv.lock` doesn't reference CUDA packages

---

### Step 4: Modify Dockerfile - Builder Stage
**Objective**: Configure UV to use system Python and respect pre-installed Intel packages

**Code Changes** (Dockerfile, builder stage around lines 142-162):

**Replace these lines** (159-162):
```dockerfile
RUN --mount=type=cache,id=ragflow_uv,target=/root/.cache/uv,sharing=locked \
    if [ "$NEED_MIRROR" == "1" ]; then \
        sed -i 's|pypi.org|mirrors.aliyun.com/pypi|g' uv.lock; \
    else \
        sed -i 's|mirrors.aliyun.com/pypi|pypi.org|g' uv.lock; \
    fi; \
    if [ "$LIGHTEN" == "1" ]; then \
        uv sync --python 3.10 --frozen; \
    else \
        uv sync --python 3.10 --frozen --all-extras; \
    fi
```

**With this**:
```dockerfile
# Configure UV to use system Python (no venv)
ENV UV_SYSTEM_PYTHON=1

# Detect Python version and sync dependencies to system
RUN --mount=type=cache,id=ragflow_uv,target=/root/.cache/uv,sharing=locked \
    PYTHON_VERSION=$(python3 --version | grep -oP '3\.\d+') && \
    echo "Using Python $PYTHON_VERSION for UV sync" && \
    if [ "$NEED_MIRROR" == "1" ]; then \
        sed -i 's|pypi.org|mirrors.aliyun.com/pypi|g' uv.lock; \
    else \
        sed -i 's|mirrors.aliyun.com/pypi|pypi.org|g' uv.lock; \
    fi; \
    if [ "$LIGHTEN" == "1" ]; then \
        uv sync --python $PYTHON_VERSION --frozen --no-install-project; \
    else \
        uv sync --python $PYTHON_VERSION --frozen --all-extras --no-install-project; \
    fi && \
    # Verify Intel PyTorch is accessible
    python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); import intel_extension_for_pytorch as ipex; print(f'IPEX version: {ipex.__version__}')"
```

**Key Changes Explained**:
- `UV_SYSTEM_PYTHON=1`: Forces UV to install to system Python (no venv)
- Dynamic Python version detection: Uses actual version from base image
- `--no-install-project`: Prevents UV from installing the project itself (done later)
- Verification step: Confirms Intel packages are importable

---

### Step 5: Modify Dockerfile - Production Stage  
**Objective**: Ensure production stage uses system Python, not venv

**Code Changes** (Dockerfile, production stage around lines 186-191):

**Remove or comment out these lines** (186-189):
```dockerfile
# Copy Python environment and packages
ENV VIRTUAL_ENV=/ragflow/.venv
COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"
```

**Replace with**:
```dockerfile
# Copy system Python packages from builder
# Intel PyTorch packages are already in base image, UV installed additional deps to system
COPY --from=builder /usr/local/lib/python3.*/site-packages /usr/local/lib/python3.*/site-packages
COPY --from=builder /usr/local/lib/python3.*/dist-packages /usr/local/lib/python3.*/dist-packages

# Ensure system Python is used
ENV PATH="/usr/local/bin:${PATH}"
```

**Additional Environment Variables** (add after line 191):
```dockerfile
# Intel XPU optimization environment variables (optional but recommended)
ENV IPEX_XPU_ONEDNN_LAYOUT=1 \
    SYCL_CACHE_PERSISTENT=1
```

---

### Step 6: Add Verification Step (Optional but Recommended)
**Objective**: Confirm Intel packages are correctly installed in final image

**Code Changes** (Dockerfile, add before ENTRYPOINT around line 213):

```dockerfile
# Verify Intel PyTorch installation
RUN python3 -c "import torch; print('PyTorch:', torch.__version__); \
    import intel_extension_for_pytorch as ipex; print('IPEX:', ipex.__version__); \
    import oneccl_bindings_for_pytorch; print('oneCCL: OK'); \
    print('XPU Available:', torch.xpu.is_available() if hasattr(torch, 'xpu') else 'Module exists')" || \
    (echo 'ERROR: Intel PyTorch verification failed' && exit 1)
```

**What this does**:
- Verifies all Intel packages are importable
- Checks versions are correct
- Fails build early if something is wrong

---

## Implementation Order

### Phase 1: Preparation (Local Machine)
1. ✅ **Backup current files**
   ```bash
   cp Dockerfile Dockerfile.backup
   cp pyproject.toml pyproject.toml.backup
   cp uv.lock uv.lock.backup
   ```

2. ✅ **Update pyproject.toml** (Step 2)
   - Add Intel index URLs
   - Update torch/Intel package versions

3. ✅ **Generate new uv.lock** (Step 3)
   ```bash
   uv lock --no-cache
   ```

### Phase 2: Dockerfile Modifications
4. ✅ **Add Python version detection** (Step 1)
   - Insert detection code in base stage

5. ✅ **Modify builder stage** (Step 4)
   - Set `UV_SYSTEM_PYTHON=1`
   - Update sync command
   - Add verification

6. ✅ **Modify production stage** (Step 5)
   - Remove venv copying
   - Copy system packages
   - Add Intel env vars

7. ✅ **Add final verification** (Step 6)
   - Confirm packages work

### Phase 3: Build and Test
8. ✅ **Build Docker image**
   ```bash
   docker build -t ragflow-intel:test .
   ```

9. ✅ **Test in container**
   ```bash
   docker run -it --rm ragflow-intel:test python3 -c "import torch; print(torch.__version__); import intel_extension_for_pytorch as ipex; print('IPEX OK')"
   ```

10. ✅ **Verify GPU recognition** (with Intel GPU access)
    ```bash
    docker run -it --rm --device=/dev/dri ragflow-intel:test python3 -c "import torch; print('XPU devices:', torch.xpu.device_count())"
    ```

---

## Success Criteria

### Build Success
- ✅ Docker build completes without errors
- ✅ No CUDA/NVIDIA package warnings during UV sync
- ✅ Intel packages verification passes in builder stage
- ✅ Final image verification passes

### Runtime Success
- ✅ `import torch` works
- ✅ `import intel_extension_for_pytorch` works
- ✅ `import oneccl_bindings_for_pytorch` works
- ✅ `torch.xpu.is_available()` returns True (with GPU)
- ✅ Application runs without import errors

---

## Troubleshooting Guide

### Issue 1: UV Lock Generation Fails
**Symptom**: `uv lock` exits with dependency resolution errors

**Solution**:
```bash
# Try with verbose output
uv lock --no-cache -v

# If Intel packages cause issues, temporarily remove from pyproject.toml
# They'll be used from base image instead
```

### Issue 2: Python Version Mismatch
**Symptom**: Build fails with "Python 3.X not found"

**Solution**:
- Check base image Python version: `docker run intel/intel-extension-for-pytorch:2.8.10-xpu python3 --version`
- Update detection logic in Dockerfile
- Ensure `pyproject.toml` has correct `requires-python`

### Issue 3: Intel Packages Not Found
**Symptom**: `ModuleNotFoundError: No module named 'intel_extension_for_pytorch'`

**Solution**:
```dockerfile
# Verify packages exist in base image
RUN python3 -c "import sys; print(sys.path)" && \
    python3 -m pip list | grep -i intel
```

### Issue 4: UV Installs CUDA Packages
**Symptom**: Build logs show CUDA package downloads

**Solution**:
- Ensure `UV_SYSTEM_PYTHON=1` is set
- Verify `uv.lock` doesn't reference CUDA
- Check Intel index URLs in `pyproject.toml`
- May need to use `--no-deps` flag for torch packages

---

## Alternative Approach (Fallback)

If system Python approach has issues, use **Option B: venv with system-site-packages**:

```dockerfile
# In builder stage, replace UV_SYSTEM_PYTHON with:
RUN uv venv --system-site-packages .venv && \
    . .venv/bin/activate && \
    uv pip install --no-deps torch torchvision torchaudio && \
    uv sync --frozen --all-extras
```

This gives venv isolation while accessing system Intel packages.

---

## File Summary

### Files to Modify
1. **pyproject.toml** - Add Intel indexes and dependencies
2. **Dockerfile** - System Python approach, remove venv
3. **uv.lock** - Regenerate with new dependencies

### Files to Backup
- Dockerfile → Dockerfile.backup
- pyproject.toml → pyproject.toml.backup  
- uv.lock → uv.lock.backup

### Files to Create
- INTEL_PYTORCH_IMPLEMENTATION_PLAN.md (this file)

---

## Risk Assessment

### Low Risk
- ✅ Using system Python in containers (Docker best practice)
- ✅ Intel packages already in base image
- ✅ Minimal changes to build process

### Medium Risk  
- ⚠️ Python version detection might need adjustment
- ⚠️ UV lock generation may have warnings (non-critical)

### Mitigation
- Test each step incrementally
- Keep backups of all modified files
- Have fallback approach ready (venv with system-site-packages)

---

## Next Steps

1. **Review this plan** with the user
2. **Get approval** to proceed
3. **Switch to Code mode** to implement changes
4. **Execute Phase 1** (local prep)
5. **Execute Phase 2** (Dockerfile mods)
6. **Execute Phase 3** (build and test)

---

## Notes

- Intel base image: `intel/intel-extension-for-pytorch:2.8.10-xpu`
- Target PyTorch: 2.8.0 (XPU build)
- IPEX version: 2.8.10+xpu
- oneCCL: 2.8.0+xpu
- Python: Likely 3.11 (to be confirmed in Step 1)

---

## References

- UV Documentation: https://docs.astral.sh/uv/
- Intel Extension for PyTorch: https://intel.github.io/intel-extension-for-pytorch/
- Docker Multi-stage Builds: https://docs.docker.com/build/building/multi-stage/