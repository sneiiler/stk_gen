"""
杂项工具函数模块

提供各种通用工具函数，包括路径处理、时间处理等功能。

Author: Yin Kaifeng <yinkaifeng@cast>
Created: 2024-06-16
"""

from pathlib import Path
import datetime
import time


def get_project_root() -> Path:
    """
    获取项目根目录路径

    Returns:
        Path: 项目根目录的Path对象

    Note:
        通过查找pyproject.toml文件来确定项目根目录
        如果找不到，则返回当前文件的上上级目录作为备选
    """
    # 从当前文件位置向上查找，直到找到包含pyproject.toml的目录
    current_dir = Path(__file__).parent
    while current_dir != current_dir.parent:
        if (current_dir / "pyproject.toml").exists():
            return current_dir
        current_dir = current_dir.parent
    return Path(__file__).parent.parent  # 如果没找到，返回当前文件的上上级目录


def ensure_dir_exists(dir_path: Path) -> None:
    """
    确保目录存在，如果不存在则创建

    Args:
        dir_path (Path): 需要确保存在的目录路径

    Returns:
        None
    """
    dir_path.mkdir(parents=True, exist_ok=True)


def get_documents_dir() -> Path:
    """
    获取项目documents目录路径

    Returns:
        Path: documents目录的Path对象
    """
    docs_dir = get_project_root() / "documents"
    ensure_dir_exists(docs_dir)
    return docs_dir

def get_data_dir() -> Path:
    """
    获取项目data目录路径
    """
    data_dir = get_project_root() / "data"
    ensure_dir_exists(data_dir)
    return data_dir


def get_current_timestamp() -> str:
    """
    获取当前时间戳字符串

    Returns:
        str: 格式为'YYYYMMDD_HHMMSS'的时间戳字符串
    """
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


# 装饰器函数
def measure_time(func):
    def wrapper(*args, **kwargs):
        # 记录开始时间
        start_time = time.time()
        # 调用原始方法
        result = func(*args, **kwargs)
        # 记录结束时间
        end_time = time.time()

        # 计算执行时间
        execution_time = end_time - start_time

        # 计算中文字符宽度
        def get_str_width(s: str) -> int:
            width = 0
            for char in s:
                # 中文字符宽度为2，其他字符宽度为1
                width += 2 if "\u4e00" <= char <= "\u9fff" else 1
            return width

        # 构建框的顶部和底部
        box_width = 50  # 设置一个合适的宽度
        top_bottom = "╔" + "═" * (box_width - 2) + "╗"
        middle = "║" + " " * (box_width - 2) + "║"
        bottom = "╚" + "═" * (box_width - 2) + "╝"

        # 构建方法名行
        method_name = func.__name__
        method_name_width = get_str_width(method_name)
        method_line = f"║ Method: {method_name}" + " " * (box_width - method_name_width - 11) + "║"

        # 构建时间行
        time_str = f"{execution_time:.4f} seconds"
        time_str_width = get_str_width(time_str)
        time_line = f"║ Time: {time_str}" + " " * (box_width - time_str_width - 9) + "║"

        # 打印完整的框
        print("\n" + top_bottom)
        print(method_line)
        print(time_line)
        print(bottom + "\n")

        return result

    return wrapper
