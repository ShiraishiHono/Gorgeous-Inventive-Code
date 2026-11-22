"""
handwritten_digit.py —— 手写数字识别（独立模块）

· 数据：MNIST（60000 张训练图 + 10000 张测试图），自动下载并做 MD5 校验
· 网络：784 → 128 → 10 全连接神经网络，纯 NumPy 实现
· 包含：矩阵运算、前向传播、Softmax 交叉熵、L2 正则、反向传播、Adam 优化、
        训练/验证划分、提前停止、测试评估、混淆矩阵、梯度检查、图片识别
· 本文件不依赖 photo_digit.py，可单独运行：python handwritten_digit.py
"""

import csv
import gzip
import hashlib
import struct
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps

# ---------------------------------------------------------------- 路径与设置
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "handwritten_data"          # 本模块专用文件夹
MODEL_PATH = DATA_DIR / "handwritten_model.npz"

LAYER_SIZES = [784, 128, 10]
EPOCHS = 30
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
L2 = 1e-4
VAL_SIZE = 5000
PATIENCE = 5
SEED = 42

MIRRORS = [
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
]
MNIST_FILES = {
    "train_images": ("train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
    "train_labels": ("train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
    "test_images": ("t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
    "test_labels": ("t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c"),
}


# ================================================================ 1. 数据下载与读取
def md5_of(path):
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(filename, expected_md5):
    """下载文件，并用 MD5 检查是否完整；失败时自动尝试下一个镜像。"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    target = DATA_DIR / filename
    if target.exists() and md5_of(target) == expected_md5:
        return target
    temp = target.with_suffix(".part")
    for mirror in MIRRORS:
        try:
            print(f"正在下载 {filename} ...")
            urllib.request.urlretrieve(mirror + filename, temp)
            if md5_of(temp) == expected_md5:
                temp.replace(target)
                return target
            print("  文件校验失败，尝试其他镜像。")
        except OSError as error:
            print(f"  下载失败：{error}")
    temp.unlink(missing_ok=True)
    raise RuntimeError(f"无法下载 {filename}，请检查网络后重试。")


def read_images(path):
    with gzip.open(path, "rb") as f:
        raw = f.read()
    magic, count, rows, cols = struct.unpack(">IIII", raw[:16])
    if magic != 2051 or rows != 28 or cols != 28:
        raise ValueError(f"{path.name} 不是有效的 MNIST 图片文件")
    pixels = np.frombuffer(raw, dtype=np.uint8, offset=16)
    if pixels.size != count * 784:
        raise ValueError(f"{path.name} 数据长度不正确")
    return pixels.reshape(count, 784).astype(np.float32) / 255.0   # 归一化到 0～1


def read_labels(path):
    with gzip.open(path, "rb") as f:
        raw = f.read()
    magic, count = struct.unpack(">II", raw[:8])
    if magic != 2049:
        raise ValueError(f"{path.name} 不是有效的 MNIST 标签文件")
    labels = np.frombuffer(raw, dtype=np.uint8, offset=8).astype(np.int64)
    if labels.size != count:
        raise ValueError(f"{path.name} 数据长度不正确")
    return labels


def load_mnist():
    paths = {key: download(*info) for key, info in MNIST_FILES.items()}
    return (read_images(paths["train_images"]), read_labels(paths["train_labels"]),
            read_images(paths["test_images"]), read_labels(paths["test_labels"]))


# ================================================================ 2. 神经网络
def log_softmax(z):
    z = z - z.max(axis=1, keepdims=True)          # 防止 exp 溢出
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


class MLP:
    """多层全连接网络：隐藏层使用 ReLU，输出层输出 10 个分数。"""

    def __init__(self, sizes, seed=0, l2=L2, dtype=np.float32):
        rng = np.random.default_rng(seed)
        self.sizes = list(sizes)
        self.l2 = l2
        self.layers = len(sizes) - 1
        self.params = {}
        for i, (n_in, n_out) in enumerate(zip(sizes[:-1], sizes[1:]), start=1):
            # He 初始化：适合 ReLU，避免信号逐层变大或消失
            self.params[f"W{i}"] = (rng.standard_normal((n_in, n_out))
                                    * np.sqrt(2.0 / n_in)).astype(dtype)
            self.params[f"b{i}"] = np.zeros(n_out, dtype=dtype)

    # ---------- 前向传播：输入 → 隐藏层 → 输出
    def forward(self, x):
        activations = [x]
        h = x
        for i in range(1, self.layers + 1):
            z = h @ self.params[f"W{i}"] + self.params[f"b{i}"]   # 矩阵乘法 + 偏置
            h = np.maximum(z, 0) if i < self.layers else z         # 隐藏层 ReLU
            activations.append(h)
        return h, activations

    def loss(self, x, y):
        logits, _ = self.forward(x)
        ce = -log_softmax(logits)[np.arange(len(y)), y].mean()
        reg = 0.5 * self.l2 * sum(np.sum(self.params[f"W{i}"] ** 2)
                                  for i in range(1, self.layers + 1))
        return float(ce + reg)

    # ---------- 反向传播：从输出层往回计算每个参数的梯度
    def loss_and_grads(self, x, y):
        logits, acts = self.forward(x)
        m = len(y)
        log_p = log_softmax(logits)
        ce = -log_p[np.arange(m), y].mean()
        reg = 0.5 * self.l2 * sum(np.sum(self.params[f"W{i}"] ** 2)
                                  for i in range(1, self.layers + 1))
        delta = np.exp(log_p)                 # Softmax 概率
        delta[np.arange(m), y] -= 1           # 概率 - 正确答案
        delta /= m
        grads = {}
        for i in range(self.layers, 0, -1):
            W = self.params[f"W{i}"]
            grads[f"W{i}"] = acts[i - 1].T @ delta + self.l2 * W
            grads[f"b{i}"] = delta.sum(axis=0)
            if i > 1:
                delta = (delta @ W.T) * (acts[i - 1] > 0)   # 经过 ReLU 的导数
        return float(ce + reg), grads

    def predict_proba(self, x):
        logits, _ = self.forward(x)
        return np.exp(log_softmax(logits))

    def save(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, sizes=np.array(self.sizes), l2=self.l2, **self.params)

    @classmethod
    def load(cls, path):
        data = np.load(path)
        model = cls(data["sizes"].tolist(), l2=float(data["l2"]))
        for key in model.params:
            model.params[key] = data[key].astype(np.float32)
        return model


class Adam:
    """Adam 优化器：根据梯度自动调整每个参数的更新步长。"""

    def __init__(self, params, lr=LEARNING_RATE, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, beta1, beta2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for k in params:
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * grads[k]
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * grads[k] ** 2
            m_hat = self.m[k] / (1 - self.b1 ** self.t)
            v_hat = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ================================================================ 3. 训练与评估
def shift_batch(x, rng, max_shift=2):
    """数据增强：整批图片随机平移几个像素，让模型更适应位置偏差。"""
    dy, dx = rng.integers(-max_shift, max_shift + 1, size=2)
    imgs = x.reshape(-1, 28, 28)
    out = np.zeros_like(imgs)
    ys, yd = slice(max(dy, 0), 28 + min(dy, 0)), slice(max(-dy, 0), 28 + min(-dy, 0))
    xs, xd = slice(max(dx, 0), 28 + min(dx, 0)), slice(max(-dx, 0), 28 + min(-dx, 0))
    out[:, ys, xs] = imgs[:, yd, xd]
    return out.reshape(-1, 784)


def evaluate(model, x, y, batch=2000):
    total_loss, correct = 0.0, 0
    preds = []
    for s in range(0, len(x), batch):
        logits, _ = model.forward(x[s:s + batch])
        log_p = log_softmax(logits)
        total_loss += -log_p[np.arange(len(log_p)), y[s:s + batch]].sum()
        p = log_p.argmax(axis=1)
        correct += (p == y[s:s + batch]).sum()
        preds.append(p)
    return total_loss / len(x), correct / len(x), np.concatenate(preds)


def train():
    x_all, y_all, x_test, y_test = load_mnist()
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(x_all))
    val_idx, train_idx = order[:VAL_SIZE], order[VAL_SIZE:]
    x_train, y_train = x_all[train_idx], y_all[train_idx]
    x_val, y_val = x_all[val_idx], y_all[val_idx]
    print(f"训练集 {len(x_train)} 张，验证集 {len(x_val)} 张，网络 {LAYER_SIZES}")

    model = MLP(LAYER_SIZES, seed=SEED)
    optimizer = Adam(model.params)
    best_ce, stale, history = float("inf"), 0, []

    for epoch in range(1, EPOCHS + 1):
        start = time.perf_counter()
        idx = rng.permutation(len(x_train))
        for s in range(0, len(idx), BATCH_SIZE):
            b = idx[s:s + BATCH_SIZE]
            xb = shift_batch(x_train[b], rng) if rng.random() < 0.5 else x_train[b]
            _, grads = model.loss_and_grads(xb, y_train[b])
            optimizer.step(model.params, grads)
        tr_ce, tr_acc, _ = evaluate(model, x_train, y_train)
        va_ce, va_acc, _ = evaluate(model, x_val, y_val)
        history.append({"epoch": epoch, "train_ce": tr_ce, "train_acc": tr_acc,
                        "val_ce": va_ce, "val_acc": va_acc})
        improved = va_ce < best_ce
        if improved:
            best_ce, stale = va_ce, 0
            model.save(MODEL_PATH)
        else:
            stale += 1
        print(f"第 {epoch:02d} 轮 | 训练准确率 {tr_acc:.2%} | 验证准确率 {va_acc:.2%} | "
              f"{time.perf_counter() - start:.1f}s" + ("  ← 保存最佳模型" if improved else ""))
        if stale >= PATIENCE:
            print(f"验证误差连续 {PATIENCE} 轮没有改善，提前停止。")
            break

    with open(DATA_DIR / "history.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    best = MLP.load(MODEL_PATH)
    test_model(best, x_test, y_test)
    return best


def test_model(model, x_test=None, y_test=None):
    if x_test is None:
        _, _, x_test, y_test = load_mnist()
    _, acc, preds = evaluate(model, x_test, y_test)
    matrix = np.zeros((10, 10), dtype=int)
    np.add.at(matrix, (y_test, preds), 1)
    print(f"\n测试集准确率：{acc:.2%}（{len(y_test)} 张）")
    print("混淆矩阵（行=真实数字，列=预测数字）：")
    print("     " + "".join(f"{i:>6}" for i in range(10)))
    for i, row in enumerate(matrix):
        print(f"{i:>4} " + "".join(f"{v:>6}" for v in row))
    with open(DATA_DIR / "confusion_matrix.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["真实\\预测"] + list(range(10)))
        for i, row in enumerate(matrix):
            writer.writerow([i] + row.tolist())


def gradient_check():
    """用数值方法验证反向传播：两种梯度应几乎相同（相对误差 < 1e-6）。"""
    rng = np.random.default_rng(0)
    model = MLP([12, 8, 5], seed=0, l2=1e-3, dtype=np.float64)
    x, y = rng.standard_normal((6, 12)), rng.integers(0, 5, 6)
    _, grads = model.loss_and_grads(x, y)
    worst, eps = 0.0, 1e-5
    for key, p in model.params.items():
        for _ in range(10):
            idx = tuple(rng.integers(0, s) for s in p.shape)
            old = p[idx]
            p[idx] = old + eps; plus = model.loss(x, y)
            p[idx] = old - eps; minus = model.loss(x, y)
            p[idx] = old
            num = (plus - minus) / (2 * eps)
            err = abs(num - grads[key][idx]) / max(1e-12, abs(num) + abs(grads[key][idx]))
            worst = max(worst, err)
    print(f"梯度检查最大相对误差：{worst:.2e}  →  {'通过' if worst < 1e-6 else '未通过'}")


# ================================================================ 4. 图片预处理与识别
def clean_path(text):
    """处理 Mac 拖入文件时带来的引号、空格转义。"""
    return Path(text.strip().strip("'\"").replace("\\ ", " ")).expanduser()


def otsu_threshold(gray_u8):
    hist = np.bincount(gray_u8.ravel(), minlength=256).astype(np.float64)
    p = hist / hist.sum()
    omega = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    sigma = (mu[-1] * omega - mu) ** 2 / (omega * (1 - omega) + 1e-12)
    return int(np.argmax(sigma))


def center_by_mass(canvas):
    total = canvas.sum()
    yy, xx = np.indices(canvas.shape)
    dy = int(round(14 - (yy * canvas).sum() / total))
    dx = int(round(14 - (xx * canvas).sum() / total))
    out = np.zeros_like(canvas)
    ys, yd = slice(max(dy, 0), 28 + min(dy, 0)), slice(max(-dy, 0), 28 + min(-dy, 0))
    xs, xd = slice(max(dx, 0), 28 + min(dx, 0)), slice(max(-dx, 0), 28 + min(-dx, 0))
    out[ys, xs] = canvas[yd, xd]
    return out


def prepare_image(path):
    """把一张手写数字照片转换成和 MNIST 一样的 784 个数：黑底白字、居中、0～1。"""
    img = ImageOps.exif_transpose(Image.open(path)).convert("L")
    img.thumbnail((800, 800))
    g = np.asarray(ImageOps.autocontrast(img), dtype=np.float32)
    border = np.concatenate([g[0], g[-1], g[:, 0], g[:, -1]])
    if np.median(border) > 127:                       # 白底黑字 → 反色成黑底白字
        g = 255 - g
        border = 255 - border
    t = otsu_threshold(g.astype(np.uint8))
    bg = np.median(border)
    fg = np.where(g > t, (g - bg) / max(g.max() - bg, 1), 0).clip(0, 1)

    rows = np.nonzero(fg.sum(axis=1) > 0.01 * fg.sum(axis=1).max())[0]
    cols = np.nonzero(fg.sum(axis=0) > 0.01 * fg.sum(axis=0).max())[0]
    if rows.size == 0 or cols.size == 0:
        raise ValueError("图片中没有找到数字")
    crop = fg[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    h, w = crop.shape
    crop_img = Image.fromarray((crop * 255).astype(np.uint8))
    if max(h, w) > 60:                                # 大图中笔画相对较细 → 适当加粗
        crop_img = crop_img.filter(ImageFilter.MaxFilter(2 * (max(h, w) // 80) + 1))
    scale = 20 / max(h, w)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    small = np.asarray(crop_img.resize((nw, nh), Image.Resampling.LANCZOS), np.float32) / 255
    canvas = np.zeros((28, 28), np.float32)
    top, left = (28 - nh) // 2, (28 - nw) // 2
    canvas[top:top + nh, left:left + nw] = small
    return center_by_mass(canvas).reshape(1, 784)


def recognize(model, path):
    x = prepare_image(path)
    probs = model.predict_proba(x)[0]
    top3 = np.argsort(probs)[::-1][:3]
    print(f"\n识别结果：{top3[0]}（可信度 {probs[top3[0]]:.1%}）")
    print("可能性前三：" + "，".join(f"{d}: {probs[d]:.1%}" for d in top3))
    if probs[top3[0]] < 0.6:
        print("提示：可信度较低，请确认图片中只有一个清晰的手写数字。")


# ================================================================ 5. 交互入口
def run():
    model = MLP.load(MODEL_PATH) if MODEL_PATH.exists() else None
    if model is None:
        print("尚未找到手写数字模型，首次使用需要训练（约数分钟）。")
        if input("现在开始训练吗？(y/n)：").strip().lower() != "y":
            return
        model = train()
    while True:
        print("\n【手写数字识别】 1 识别图片  2 重新训练  3 测试准确率  4 梯度检查  0 返回")
        choice = input("请选择：").strip()
        try:
            if choice == "1":
                path = clean_path(input("请输入或拖入图片路径："))
                recognize(model, path)
            elif choice == "2":
                model = train()
            elif choice == "3":
                test_model(model)
            elif choice == "4":
                gradient_check()
            elif choice == "0":
                return
            else:
                print("请输入 0～4。")
        except (OSError, ValueError, RuntimeError) as error:
            print(f"出错：{error}")


if __name__ == "__main__":
    run()
