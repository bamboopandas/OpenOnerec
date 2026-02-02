import os
from huggingface_hub import snapshot_download

# ================= 配置区域 =================

# 1. 代理设置 (如果不使用代理，请留空 "")
# 常见的本地代理地址: "http://127.0.0.1:7890" 或 "socks5://127.0.0.1:7890"
PROXY_URL = ""

# 2. 是否使用国内镜像 (建议开启，速度快且稳定，无需代理)
USE_MIRROR = True

# 3. HuggingFace Token (如果下载受限数据集，请填入 Token)
# 获取地址: https://huggingface.co/settings/tokens
HF_TOKEN = "YOUR_HF_TOKEN_HERE"

# ===========================================

# 设置环境变量
if USE_MIRROR:
    print("🚀 使用 HF-Mirror 镜像源加速...")
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
elif PROXY_URL:
    print(f"🌐 使用代理: {PROXY_URL}")
    os.environ["https_proxy"] = PROXY_URL
    os.environ["http_proxy"] = PROXY_URL

# 数据集列表
datasets = [
    ("OpenOneRec/OpenOneRec-General-Pretrain", "./raw_data/general_text/pretrain"),
    ("OpenOneRec/OpenOneRec-General-SFT", "./raw_data/general_text/sft"),
    ("OpenOneRec/OpenOneRec-RecIF", "./raw_data/onerec_data"),
]

print("📦 开始下载数据...")

for repo_id, local_dir in datasets:
    print(f"\n⬇️  正在下载: {repo_id} -> {local_dir}")
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
            token=HF_TOKEN if HF_TOKEN else None,
            # max_workers=8 # 如果带宽够大，可以取消注释开启多线程
        )
        print(f"✅ 下载成功: {repo_id}")
    except Exception as e:
        print(f"❌ 下载失败 {repo_id}: {str(e)}")
        print("提示: 请检查网络连接，或尝试配置 HF_TOKEN")

print("\n🎉 所有任务已结束。")
