import os
import sys
from huggingface_hub import snapshot_download

# ================= 配置区域 =================
MODEL_ID = "Qwen/Qwen3-0.6B"
LOCAL_DIR = "/zhdd/home/lkzhang/vscode/evaluate_exp/OpenOneRec/code/onerec_pretrain/hf_models/Qwen3-0.6B"
HF_MIRROR_URL = "https://hf-mirror.com"
PROXY_URL = "http://192.168.1.99:7890"
# ===========================================

def download_model(use_proxy=False):
    if use_proxy:
        os.environ["http_proxy"] = PROXY_URL
        os.environ["https_proxy"] = PROXY_URL
        print(f"🔌 [模式] 代理模式: {PROXY_URL}")
    else:
        for k in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY']:
            if k in os.environ: del os.environ[k]
        print(f"🛡️ [模式] 直连模式 (已清除代理环境变量)")

    os.environ["HF_ENDPOINT"] = HF_MIRROR_URL
    print(f"🎯 目标镜像: {HF_MIRROR_URL}")

    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=LOCAL_DIR,
        local_dir_use_symlinks=False,
        resume_download=True,
        etag_timeout=60
    )

print(f"🚀 开始下载基座模型: {MODEL_ID}")

# 第一次尝试：直连
try:
    print("\n>>> 尝试 1: 直连镜像站...")
    download_model(use_proxy=False)
    print("✅ 下载成功！")
    sys.exit(0)
except Exception as e:
    print(f"⚠️ 直连失败: {str(e)}")

# 第二次尝试：代理
try:
    print("\n>>> 尝试 2: 挂代理连接镜像站...")
    download_model(use_proxy=True)
    print("✅ 下载成功！")
    sys.exit(0)
except Exception as e:
    print(f"❌ 代理连接也失败: {str(e)}")
