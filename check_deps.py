import os
import re
import sys
import subprocess
import pkg_resources

# 定义要扫描的目录（递归）
SCAN_DIRS = ["pretrain/recipes", "pretrain/onerec_llm", "tokenizer"]

# 定义已知的内置库（不需要安装的）
BUILTIN_MODULES = sys.builtin_module_names
STD_LIB = {
    'os', 'sys', 're', 'json', 'math', 'random', 'time', 'datetime', 'logging', 
    'argparse', 'collections', 'itertools', 'functools', 'pathlib', 'typing', 
    'copy', 'shutil', 'subprocess', 'glob', 'pickle', 'warnings', 'contextlib',
    'abc', 'io', 'gc', 'platform', 'threading', 'multiprocessing', 'queue', 'traceback'
}

def get_imports_from_file(filepath):
    imports = set()
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # 匹配 import xxx
    matches = re.findall(r'^\s*import\s+([\w\.]+)', content, re.MULTILINE)
    for m in matches:
        imports.add(m.split('.')[0])
        
    # 匹配 from xxx import yyy
    matches = re.findall(r'^\s*from\s+([\w\.]+)\s+import', content, re.MULTILINE)
    for m in matches:
        imports.add(m.split('.')[0])
        
    return imports

def get_installed_packages():
    return {pkg.key for pkg in pkg_resources.working_set}

def main():
    print("🔍 开始扫描代码中的依赖...")
    required_modules = set()
    
    for d in SCAN_DIRS:
        if not os.path.exists(d): continue
        for root, _, files in os.walk(d):
            for file in files:
                if file.endswith(".py"):
                    imports = get_imports_from_file(os.path.join(root, file))
                    required_modules.update(imports)

    # 过滤掉项目自己的模块（假设以 onerec_llm 开头或在当前目录下的文件夹名）
    local_modules = {'onerec_llm', 'recipes', 'tools', 'utils', 'dataset', 'module', 'model'}
    
    # 过滤掉标准库
    filtered_modules = {
        m for m in required_modules 
        if m not in BUILTIN_MODULES 
        and m not in STD_LIB 
        and m not in local_modules
    }

    print(f"📦 代码中检测到的第三方库: {sorted(filtered_modules)}")
    
    print("\n🔍 正在检查当前环境...")
    installed = get_installed_packages()
    
    # 映射表：有些库 import 名字和 pip 安装名不一样
    # key: import名, value: pip名
    MAPPING = {
        'PIL': 'Pillow',
        'sklearn': 'scikit-learn',
        'yaml': 'PyYAML',
        'cv2': 'opencv-python',
        'faiss': 'faiss-gpu', # 或 faiss-cpu
        'tensorboard': 'tensorboard',
        'torch': 'torch',
        'transformers': 'transformers',
        'torchdata': 'torchdata'
    }

    missing = []
    for module in filtered_modules:
        # 检查是否已安装
        package_name = MAPPING.get(module, module)
        
        # 模糊匹配：比如 torch 在环境里可能叫 torch-2.0...
        # 这里简单判断 package_name 是否在 installed 集合里
        # 注意：pkg_resources 的 key 都是小写的
        if package_name.lower() not in installed:
            # 二次检查：有些库可能已经以 import 名字安装了
            if module.lower() not in installed:
                missing.append(package_name)

    print("\n" + "="*40)
    if missing:
        print(f"❌ 发现 {len(missing)} 个缺失的库:")
        for m in missing:
            print(f"  - {m}")
        
        print("\n💡 建议运行以下命令安装:")
        print(f"pip install {' '.join(missing)} -i https://pypi.tuna.tsinghua.edu.cn/simple")
    else:
        print("✅ 恭喜！主要依赖看似都已安装。")
        print("(注：这只是静态扫描，某些动态加载的库可能未被检测到)")

if __name__ == "__main__":
    main()
