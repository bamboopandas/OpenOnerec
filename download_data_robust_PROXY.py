import os
import sys
from huggingface_hub import snapshot_download

# ================= 配置区域 =================
# 请务必替换为您的真实 Token
HF_TOKEN = "YOUR_HF_TOKEN_HERE" 
HF_MIRROR_URL = "https://hf-mirror.com"
PROXY_URL = "http://192.168.1.99:7890"
# ===========================================

dataset_id = "OpenOneRec/OpenOneRec-RecIF"
local_dir = "./raw_data/onerec_data"

def download_task(use_proxy=False):
    # 环境变量管理
    if use_proxy:
        os.environ["http_proxy"] = PROXY_URL
        os.environ["https_proxy"] = PROXY_URL
        print(f"🔌 [模式] 代理模式: {PROXY_URL}")
    else:
        for k in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY']:
            if k in os.environ: del os.environ[k]
        print(f"🛡️ [模式] 直连模式 (已清除代理环境变量)")

    # 始终使用镜像站
    os.environ["HF_ENDPOINT"] = HF_MIRROR_URL
    print(f"🎯 目标: {HF_MIRROR_URL}")

    snapshot_download(
        repo_id=dataset_id,
        repo_type="dataset",
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
        token=HF_TOKEN,
        etag_timeout=60
    )

if HF_TOKEN.startswith("在此"):
    print("❌ 错误：请先编辑此文件填入您的 HF_TOKEN！")
    sys.exit(1)

print(f"📦 准备下载: {dataset_id}")

# 第一次尝试：直连镜像站
try:
    print("\n>>> 尝试 1: 直连镜像站 (推荐)...")
    download_task(use_proxy=False)
    print("✅ 下载成功！")
    sys.exit(0)
except Exception as e:
    print(f"⚠️ 直连失败: {str(e)}")

# 第二次尝试：代理连接镜像站
try:
    print("\n>>> 尝试 2: 挂代理连接镜像站 (备选)...")
    download_task(use_proxy=True)
    print("✅ 下载成功！")
    sys.exit(0)
except Exception as e:
    print(f"❌ 代理连接也失败: {str(e)}")
    print("请检查 Token 权限或网络连通性。")
