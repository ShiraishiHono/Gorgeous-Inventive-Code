"""
main.py —— 数字识别程序启动入口

运行后询问识别类型：
    0 = 识别照片中的数字（印刷体，如门牌、价签、书本上的数字）
    1 = 识别手写数字（基于 MNIST 训练）

两套识别程序分别位于：
    photo_digit.py        照片数字识别（完全独立）
    handwritten_digit.py  手写数字识别（完全独立）

两个文件互不导入、互不共用代码和数据文件夹。
删除其中一个文件后，另一个仍可通过本入口或单独运行正常使用。
"""

import importlib.util
import sys
from pathlib import Path

# main.py 所在的文件夹；识别程序必须和 main.py 放在同一个文件夹
BASE_DIR = Path(__file__).resolve().parent

# 选项 → (模块文件名, 显示名称)
RECOGNIZERS = {
    "0": ("photo_digit", "照片数字识别"),
    "1": ("handwritten_digit", "手写数字识别"),
}


def ask_choice():
    """反复询问，直到用户输入 0 或 1；输入 q 退出。"""
    while True:
        print("\n请选择需要识别的数字类型：")
        print("  0 —— 识别照片中的数字（印刷体）")
        print("  1 —— 识别手写数字")
        print("  q —— 退出程序")
        choice = input("请输入 0 或 1：").strip().lower()
        if choice in RECOGNIZERS or choice == "q":
            return choice
        print("输入无效，请只输入 0 或 1。")


def load_recognizer(module_name):
    """按文件路径导入 main.py 旁边的识别程序；文件不存在时返回 None。"""
    file_path = BASE_DIR / f"{module_name}.py"
    if not file_path.is_file():
        return None
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as error:
        del sys.modules[module_name]
        print(f"缺少依赖库：{error.name}。请先在终端运行：pip install numpy pillow")
        sys.exit(1)
    return module


def main():
    print("=" * 50)
    print("        数字识别程序（NumPy 神经网络）")
    print("=" * 50)
    while True:
        choice = ask_choice()
        if choice == "q":
            print("程序已退出。")
            return
        module_name, title = RECOGNIZERS[choice]
        module = load_recognizer(module_name)
        if module is None:
            print(f"未找到 {module_name}.py，“{title}”功能不可用。")
            print(f"程序查找的文件夹是：{BASE_DIR}")
            others = sorted(p.name for p in BASE_DIR.iterdir() if p.is_file())
            print(f"该文件夹中现有文件：{', '.join(others) or '（空）'}")
            continue
        print(f"\n—— 进入：{title} ——")
        module.run()


if __name__ == "__main__":
    main()
