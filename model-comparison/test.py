import matplotlib
import matplotlib.font_manager as fm
import os
import shutil

print("=== 清除Matplotlib字体缓存 ===")

# 1. 找到缓存目录
try:
    cache_dir = matplotlib.get_cachedir()
except:
    # 备用方法
    import tempfile

    cache_dir = os.path.join(tempfile.gettempdir(), 'matplotlib')

print(f"缓存目录: {cache_dir}")

# 2. 删除缓存文件（不是整个目录）
if os.path.exists(cache_dir):
    # 只删除字体缓存文件
    cache_files = [
        os.path.join(cache_dir, 'fontlist-v330.json'),
        os.path.join(cache_dir, 'fontlist*.json'),
        os.path.join(cache_dir, '*.cache'),
    ]

    import glob

    for pattern in cache_files:
        for file in glob.glob(pattern):
            try:
                os.remove(file)
                print(f"已删除: {file}")
            except:
                pass

# 3. 重建缓存（不同版本的方法）
print("正在重建字体缓存...")
try:
    # 方法1：新版本
    fm._rebuild()
except AttributeError:
    try:
        # 方法2：另一种重建方式
        fm.fontManager._rebuild()
    except:
        try:
            # 方法3：直接重新初始化
            fm._load_fontmanager(try_read_cache=False)
        except:
            # 方法4：删除缓存后重启内核最有效
            print("⚠️ 需要重启Jupyter内核来生效")
            print("请运行: Kernel -> Restart Kernel")

# 4. 设置英文字体
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif'],
    'axes.unicode_minus': False
})

print("✅ 字体配置已更新")