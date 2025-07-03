# data_manager.py
import os
from pathlib import Path
import time
from flask import jsonify
from werkzeug.utils import secure_filename

DATA_ROOT = Path(__file__).parent / '../data'
print(DATA_ROOT)

def get_file_stats():
    """获取数据目录的完整统计信息"""
    categories = ['raw', 'processed', 'predictions']
    stats = {cat: {'count': 0, 'total_size': 0, 'files': []} for cat in categories}
    
    for category in categories:
        dir_path = DATA_ROOT / category
        if not dir_path.exists():
            continue
            
        for file in dir_path.iterdir():
            if file.is_file() and identify_csv(file.name):
                stat = file.stat()
                stats[category]['files'].append({
                    'name': file.name,
                    'size': _human_readable_size(stat.st_size),
                    'bytes': stat.st_size,
                    'modified': time.strftime('%Y-%m-%d %H:%M', time.localtime(stat.st_mtime)),
                    'category': category,
                    'path': str(file.relative_to(DATA_ROOT))
                })
                stats[category]['count'] += 1
                stats[category]['total_size'] += stat.st_size
                
        # 转换总大小为可读格式
        stats[category]['total_size'] = _human_readable_size(stats[category]['total_size'])
        
    return stats

def delete_file(relative_path):
    """安全删除文件"""
    target = DATA_ROOT / relative_path
    if not target.exists():
        return False, "文件不存在"
    if not target.is_file():
        return False, "路径不是文件"
    if target.parent.name not in ['raw', 'processed', 'predictions']:
        return False, "非法目录操作"
    
    try:
        target.unlink()
        return True, "删除成功"
    except Exception as e:
        return False, f"删除失败: {str(e)}"

def save_uploaded_file(file, category):
    """保存上传文件到指定分类"""
    allowed_categories = ['raw', 'processed']
    if category not in allowed_categories:
        return False, "无效的分类目录"
    
    save_dir = DATA_ROOT / category
    save_dir.mkdir(exist_ok=True)
    
    # 安全文件名处理
    filename = secure_filename(file.filename)
    if not filename.lower().endswith('.csv'):
        return False, "仅支持CSV文件"
    
    target_path = save_dir / filename
    if target_path.exists():
        return False, "文件已存在"
    
    try:
        file.save(target_path)
        return True, filename
    except Exception as e:
        return False, f"保存失败: {str(e)}"

def _human_readable_size(size_bytes):
    """转换字节数为易读格式"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f}GB"

def identify_csv(file_name):
    """判断文件是否为CSV文件"""
    return file_name.lower().endswith('.csv')

# a=get_file_stats()
# print(a)