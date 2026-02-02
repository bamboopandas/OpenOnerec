import os
import sys
from huggingface_hub import snapshot_download

# ================= Configuration =================
# Defaulting to 1.7B model. Change to "OpenOneRec/OneRec-8B" if needed.
MODEL_ID = "OpenOneRec/OneRec-8B" 
# Saving to a clear 'checkpoints' directory in the root
LOCAL_DIR = os.path.join(os.getcwd(), "checkpoints", "OneRec-8B")

HF_MIRROR_URL = "https://hf-mirror.com"
PROXY_URL = "http://192.168.1.99:7890"
# ===============================================

def download_model(use_proxy=False):
    if use_proxy:
        os.environ["http_proxy"] = PROXY_URL
        os.environ["https_proxy"] = PROXY_URL
        print(f"🔌 [Mode] Proxy: {PROXY_URL}")
    else:
        for k in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY']:
            if k in os.environ: del os.environ[k]
        print(f"🛡️ [Mode] Direct Connection (Proxy env vars cleared)")

    os.environ["HF_ENDPOINT"] = HF_MIRROR_URL
    print(f"🎯 Mirror: {HF_MIRROR_URL}")
    print(f"📂 Destination: {LOCAL_DIR}")

    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=LOCAL_DIR,
        local_dir_use_symlinks=False,
        resume_download=True,
        etag_timeout=60
    )

print(f"🚀 Starting download for: {MODEL_ID}")

# Attempt 1: Direct
try:
    print("\n>>> Attempt 1: Direct connection to mirror...")
    download_model(use_proxy=False)
    print("✅ Download successful!")
    sys.exit(0)
except Exception as e:
    print(f"⚠️ Direct connection failed: {str(e)}")

# Attempt 2: Proxy
try:
    print("\n>>> Attempt 2: Using proxy...")
    download_model(use_proxy=True)
    print("✅ Download successful!")
    sys.exit(0)
except Exception as e:
    print(f"❌ Proxy connection also failed: {str(e)}")
