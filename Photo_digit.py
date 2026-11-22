"""
photo_digit.py —— 照片中的手写数字识别（完全独立，可单独运行）

使用场景：
    在纸上用笔写数字 → 用手机拍照 → 把照片交给程序 → 程序识别出照片里写的数字
    支持一张照片里同一行写多个数字（例如 "2025"）

整体流程：
    1. 训练：用 MNIST 手写数字数据训练神经网络，训练时对图片做随机
       旋转、缩放、平移、笔画粗细、模糊、噪点等“数据增强”，模拟真实拍照的效果
    2. 照片预处理：自动摆正方向 → 转灰度 → 估计纸张背景并去除光照不均
       → 自动阈值找出笔迹 → 连通区域分割出每个数字 → 调整成 MNIST 同款 28×28 格式
    3. 神经网络识别：正向传播得到 0~9 的概率，取最大者

神经网络：784 → 256 → 128 → 10，ReLU + Softmax，交叉熵 + L2 正则，Adam 优化
全部用 NumPy 矩阵运算手写实现（正向传播、反向传播、梯度检查）

依赖：pip install numpy pillow        （识别 iPhone 的 HEIC 照片还需 pip install pillow-heif）
数据和模型保存在本文件旁边的 photo_data/ 文件夹，不与其他程序共用。
"""

import gzip
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps

try:  # 可选：支持 iPhone 的 HEIC 照片
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

# 兼容新旧版本 Pillow 的常量写法
AFFINE = getattr(Image, "Transform", Image).AFFINE
BILINEAR = getattr(Image, "Resampling", Image).BILINEAR
LANCZOS = getattr(Image, "Resampling", Image).LANCZOS

DATA_DIR = Path(__file__).resolve().parent / "photo_data"
MODEL_PATH = DATA_DIR / "photo_model.npz"
DEBUG_PATH = DATA_DIR / "last_input.png"

MIRRORS = [
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
]
FILES = {
    "train_x": "train-images-idx3-ubyte.gz",
    "train_y": "train-labels-idx1-ubyte.gz",
    "test_x": "t10k-images-idx3-ubyte.gz",
    "test_y": "t10k-labels-idx1-ubyte.gz",
}
LAYERS = [784, 256, 128, 10]

# 部分 Mac 上的 NumPy（Apple Accelerate 加速库）在矩阵乘法时会误报
# “divide by zero / overflow encountered in matmul”，结果其实是正确的。
# 这里关闭这类误报；真正出问题时（损失变成 nan），训练函数会主动停止并提示。
np.seterr(divide="ignore", over="ignore", invalid="ignore")


# =====================================================================
# 1. 数据：下载并读取 MNIST
# =====================================================================
def download(name):
    path = DATA_DIR / name
    if path.exists():
        return path
    DATA_DIR.mkdir(exist_ok=True)
    for base in MIRRORS:
        try:
            print(f"正在下载 {name} ...")
            urllib.request.urlretrieve(base + name, path)
            return path
        except Exception as error:
            print(f"  从 {base} 下载失败：{error}")
            if path.exists():
                path.unlink()
    raise RuntimeError(f"无法下载 {name}，请检查网络连接。")


def load_images(name):
    with gzip.open(download(name), "rb") as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    return data.reshape(-1, 28, 28)


def load_labels(name):
    with gzip.open(download(name), "rb") as f:
        return np.frombuffer(f.read(), np.uint8, offset=8).astype(np.int64)


def load_mnist():
    return (load_images(FILES["train_x"]), load_labels(FILES["train_y"]),
            load_images(FILES["test_x"]), load_labels(FILES["test_y"]))


# =====================================================================
# 2. 数据增强：让干净的 MNIST 图片更像“拍照拍出来的”手写数字
# =====================================================================
def augment(batch_u8, rng):
    out = np.empty((len(batch_u8), 784), np.float32)
    for i, arr in enumerate(batch_u8):
        img = Image.fromarray(arr)
        # 随机旋转 ±15°、缩放 0.8~1.15 倍、平移 ±2.5 像素（仿射变换）
        theta = np.deg2rad(rng.uniform(-15, 15))
        s = rng.uniform(0.8, 1.15)
        dx, dy = rng.uniform(-2.5, 2.5, size=2)
        cos, sin = np.cos(theta), np.sin(theta)
        a, b, d, e = cos / s, sin / s, -sin / s, cos / s
        c = 14 - a * (14 + dx) - b * (14 + dy)
        f = 14 - d * (14 + dx) - e * (14 + dy)
        img = img.transform((28, 28), AFFINE, (a, b, c, d, e, f), resample=BILINEAR)
        # 随机改变笔画粗细（模拟不同的笔）
        r = rng.random()
        if r < 0.25:
            img = img.filter(ImageFilter.MaxFilter(3))
        elif r < 0.35:
            img = img.filter(ImageFilter.MinFilter(3))
        # 随机模糊（模拟对焦不准）
        if rng.random() < 0.3:
            img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 0.9)))
        x = np.asarray(img, np.float32) / 255.0
        # 随机噪点（模拟纸张纹理、拍照噪声）
        if rng.random() < 0.5:
            x = x + rng.normal(0, 0.06, x.shape).astype(np.float32)
        out[i] = np.clip(x, 0, 1).ravel()
    return out


# =====================================================================
# 3. 神经网络：矩阵运算、正向传播、反向传播、Adam
# =====================================================================
def init_params(sizes, seed=0):
    """He 初始化：W 形状 (输入, 输出)，b 初始为 0。"""
    rng = np.random.default_rng(seed)
    params = {}
    for i in range(len(sizes) - 1):
        params[f"W{i + 1}"] = rng.normal(0, np.sqrt(2 / sizes[i]), (sizes[i], sizes[i + 1]))
        params[f"b{i + 1}"] = np.zeros(sizes[i + 1])
    return params


def softmax(Z):
    Z = Z - Z.max(axis=1, keepdims=True)  # 防止 exp 溢出
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


def forward(params, X):
    """正向传播：每层做 Z = A·W + b，隐藏层用 ReLU，输出层用 Softmax。"""
    n = len(params) // 2
    cache = {"A0": X}
    A = X
    for i in range(1, n + 1):
        Z = A @ params[f"W{i}"] + params[f"b{i}"]
        A = np.maximum(Z, 0) if i < n else softmax(Z)
        cache[f"Z{i}"], cache[f"A{i}"] = Z, A
    return A, cache


def loss_fn(params, probs, y, l2):
    """交叉熵 + L2 正则。"""
    m = len(y)
    ce = -np.log(probs[np.arange(m), y] + 1e-12).mean()
    reg = 0.5 * l2 * sum(np.sum(v ** 2) for k, v in params.items() if k.startswith("W"))
    return ce + reg


def backward(params, cache, y, l2):
    """反向传播：从输出层开始，逐层算出每个参数的梯度。"""
    n = len(params) // 2
    m = len(y)
    grads = {}
    dZ = cache[f"A{n}"].copy()
    dZ[np.arange(m), y] -= 1          # Softmax + 交叉熵 的梯度：预测 - 真实
    dZ /= m
    for i in range(n, 0, -1):
        grads[f"W{i}"] = cache[f"A{i - 1}"].T @ dZ + l2 * params[f"W{i}"]
        grads[f"b{i}"] = dZ.sum(axis=0)
        if i > 1:
            dZ = (dZ @ params[f"W{i}"].T) * (cache[f"Z{i - 1}"] > 0)  # 经过 ReLU 传回上一层
    return grads


class Adam:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps, self.t = lr, b1, b2, eps, 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params, grads):
        self.t += 1
        for k in params:
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * grads[k]
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * grads[k] ** 2
            m_hat = self.m[k] / (1 - self.b1 ** self.t)
            v_hat = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def predict(params, X, batch=1000):
    return np.vstack([forward(params, X[i:i + batch])[0] for i in range(0, len(X), batch)])


def accuracy(params, X, y):
    return float((predict(params, X).argmax(axis=1) == y).mean())


# =====================================================================
# 4. 训练、保存、读取
# =====================================================================
def save_model(params):
    DATA_DIR.mkdir(exist_ok=True)
    np.savez(MODEL_PATH, **params)


def load_model():
    """读取模型；文件不存在或结构不符（例如旧版本模型）时返回 None。"""
    if not MODEL_PATH.exists():
        return None
    try:
        data = np.load(MODEL_PATH)
        params = {k: data[k] for k in data.files}
        for i in range(len(LAYERS) - 1):
            if params[f"W{i + 1}"].shape != (LAYERS[i], LAYERS[i + 1]):
                return None
        return params
    except Exception:
        return None


def train(epochs=12, batch=128, lr=1e-3, l2=1e-4):
    train_x, train_y, _, _ = load_mnist()
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(train_x))
    val_idx, tr_idx = idx[:5000], idx[5000:]          # 5000 张验证，55000 张训练
    X_val = train_x[val_idx].reshape(-1, 784) / 255.0
    y_val = train_y[val_idx]

    params = init_params(LAYERS)
    opt = Adam(params, lr)
    best = 0.0
    print(f"开始训练：共 {epochs} 轮，每轮 {len(tr_idx)} 张（带数据增强），需要几分钟……")
    for epoch in range(1, epochs + 1):
        start = time.time()
        order = rng.permutation(tr_idx)
        total_loss = 0.0
        for s in range(0, len(order), batch):
            bi = order[s:s + batch]
            X_b, y_b = augment(train_x[bi], rng), train_y[bi]
            probs, cache = forward(params, X_b)
            batch_loss = loss_fn(params, probs, y_b, l2)
            if not np.isfinite(batch_loss):
                raise RuntimeError("训练出现数值错误（损失变成 nan），请把学习率 lr 调小后重新训练。")
            total_loss += batch_loss * len(bi)
            opt.step(params, backward(params, cache, y_b, l2))
        if epoch == int(epochs * 0.7):
            opt.lr *= 0.3                              # 后期降低学习率，微调
        acc = accuracy(params, X_val, y_val)
        mark = ""
        if acc > best:
            best = acc
            save_model(params)
            mark = "  ← 已保存"
        print(f"第 {epoch:2d} 轮  损失 {total_loss / len(order):.4f}  "
              f"验证准确率 {acc:.2%}  用时 {time.time() - start:.1f}s{mark}")
    print(f"训练完成，最佳验证准确率 {best:.2%}")
    return load_model()


# =====================================================================
# 5. 照片预处理：从照片中找出每个手写数字，并转换成 28×28
# =====================================================================
def open_photo(path):
    img = ImageOps.exif_transpose(Image.open(path))   # 按手机拍摄方向摆正
    img = img.convert("L")                              # 转灰度
    scale = 800 / max(img.size)
    if scale < 1:                                       # 缩小大照片，加快处理
        img = img.resize((round(img.width * scale), round(img.height * scale)), LANCZOS)
    return img


def ink_map(gray_img):
    """估计纸张背景亮度，用“背景 - 像素”得到笔迹浓度，去除阴影和光照不均。"""
    gray = np.asarray(gray_img, np.float64)
    w, h = gray_img.size
    small = gray_img.resize((max(1, w // 4), max(1, h // 4)), BILINEAR)
    small = small.filter(ImageFilter.MaxFilter(5))      # 去掉细笔画，只留纸的亮度
    small = small.filter(ImageFilter.GaussianBlur(max(w, h) / 80))
    bg = np.asarray(small.resize((w, h), BILINEAR), np.float64)
    if np.median(gray) < 110:                           # 深色背景、浅色字的情况
        return np.clip(gray - (255 - bg), 0, None)
    return np.clip(bg - gray, 0, None)


def otsu(values):
    """大津法：自动找出区分“笔迹”和“纸”的最佳阈值。"""
    v = np.clip(values, 0, 255).astype(np.uint8)
    hist = np.bincount(v.ravel(), minlength=256).astype(np.float64)
    p = hist / hist.sum()
    omega = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    sigma = (mu[-1] * omega - mu) ** 2 / (omega * (1 - omega) + 1e-12)
    return int(np.argmax(sigma))


def find_components(mask):
    """连通区域：把连在一起的笔迹像素归为一组（8 邻域）。"""
    H, W = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for y0, x0 in zip(*np.nonzero(mask)):
        if seen[y0, x0]:
            continue
        seen[y0, x0] = True
        stack, ys, xs = [(y0, x0)], [], []
        while stack:
            y, x = stack.pop()
            ys.append(y)
            xs.append(x)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
        comps.append((np.array(ys), np.array(xs)))
    return comps


def segment(mask):
    """分割出每个数字，按从左到右排序。返回 [(ys, xs), ...]。"""
    H, W = mask.shape
    groups = []
    for ys, xs in find_components(mask):
        if len(ys) < 15:
            continue                                    # 太小：噪点
        if ys.min() == 0 or xs.min() == 0 or ys.max() == H - 1 or xs.max() == W - 1:
            continue                                    # 碰到照片边缘：多半是纸边或阴影
        if xs.max() - xs.min() > 0.8 * W:
            continue                                    # 太宽：多半是横线
        groups.append([ys, xs])
    if not groups:
        return []

    # 合并左右范围大量重叠的部分（例如断开的“5”的横杠、“4”的两笔）
    groups.sort(key=lambda g: g[1].min())
    merged = [groups[0]]
    for ys, xs in groups[1:]:
        pys, pxs = merged[-1]
        overlap = min(pxs.max(), xs.max()) - max(pxs.min(), xs.min())
        narrow = min(pxs.max() - pxs.min(), xs.max() - xs.min()) + 1
        if overlap >= 0.5 * narrow:
            merged[-1] = [np.concatenate([pys, ys]), np.concatenate([pxs, xs])]
        else:
            merged.append([ys, xs])

    # 去掉明显比最高数字矮很多的碎片（小点、污渍）
    max_h = max(g[0].max() - g[0].min() + 1 for g in merged)
    return [g for g in merged if g[0].max() - g[0].min() + 1 >= 0.35 * max_h]


def to_mnist(ys, xs):
    """把一个数字的笔迹转换成 MNIST 格式：黑底白字、20×20 内、按重心居中到 28×28。"""
    h, w = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
    pad = max(h, w) // 5 + 2
    patch = np.zeros((h + 2 * pad, w + 2 * pad), bool)
    patch[ys - ys.min() + pad, xs - xs.min() + pad] = True

    # 估计笔画粗细，太细就加粗到接近 MNIST 的比例（约为字高的 12%）
    inner = patch[1:-1, 1:-1]
    interior = inner & patch[:-2, 1:-1] & patch[2:, 1:-1] & patch[1:-1, :-2] & patch[1:-1, 2:]
    perimeter = max(1, int((inner & ~interior).sum()))
    stroke = 2 * patch.sum() / perimeter
    k = int(round(0.12 * max(h, w) - stroke))
    img = Image.fromarray(patch.astype(np.uint8) * 255)
    if k >= 2:
        img = img.filter(ImageFilter.MaxFilter(min(k | 1, 15)))

    arr = np.asarray(img, np.float64)
    r, c = np.nonzero(arr)
    arr = arr[r.min():r.max() + 1, c.min():c.max() + 1]
    h, w = arr.shape
    s = 20 / max(h, w)
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    small = Image.fromarray(arr.astype(np.uint8)).resize((nw, nh), LANCZOS)
    canvas = np.zeros((28, 28))
    top, left = (28 - nh) // 2, (28 - nw) // 2
    canvas[top:top + nh, left:left + nw] = np.clip(np.asarray(small, np.float64), 0, 255)

    total = canvas.sum()
    if total > 0:                                       # 按重心移动到中心
        cy = (canvas.sum(axis=1) * np.arange(28)).sum() / total
        cx = (canvas.sum(axis=0) * np.arange(28)).sum() / total
        canvas = np.roll(canvas, (int(round(14 - cy)), int(round(14 - cx))), axis=(0, 1))
    return canvas / max(canvas.max(), 1e-9)


def save_debug(X):
    """把送进网络的 28×28 图片拼起来放大保存，方便演示“网络看到了什么”。"""
    strip = np.hstack([np.pad(x.reshape(28, 28), 2, constant_values=0.3) for x in X])
    img = Image.fromarray((strip * 255).astype(np.uint8))
    img.resize((img.width * 6, img.height * 6), Image.NEAREST if not hasattr(Image, "Resampling")
               else Image.Resampling.NEAREST).save(DEBUG_PATH)


def recognize(params, path):
    gray = open_photo(path)
    ink = ink_map(gray)
    mask = ink > max(otsu(ink), 15)
    if mask.mean() > 0.3:
        print("提示：照片中深色区域太多，可能光线太暗或背景不是白纸，结果可能不准。")
    groups = segment(mask)
    if not groups:
        print("没有在照片中找到数字。请用深色笔写在白纸上，拍清楚一些再试。")
        return
    X = np.stack([to_mnist(ys, xs).ravel() for ys, xs in groups])
    probs = predict(params, X)
    save_debug(X)

    digits = []
    for i, p in enumerate(probs, 1):
        d = int(p.argmax())
        digits.append(str(d))
        note = "（不太确定）" if p[d] < 0.6 else ""
        second = int(np.argsort(p)[-2])
        print(f"  第 {i} 个数字：{d}   置信度 {p[d]:.1%}   第二可能：{second}（{p[second]:.1%}）{note}")
    print(f"\n识别结果：{''.join(digits)}")
    print(f"网络实际看到的图片已保存到：{DEBUG_PATH}")


# =====================================================================
# 6. 梯度检查：验证反向传播算得对不对
# =====================================================================
def gradient_check(eps=1e-5, l2=1e-3):
    rng = np.random.default_rng(0)
    params = init_params([6, 5, 4, 3], seed=1)
    X = rng.normal(size=(4, 6))
    y = rng.integers(0, 3, 4)
    probs, cache = forward(params, X)
    grads = backward(params, cache, y, l2)
    worst = 0.0
    for k in params:
        numeric = np.zeros_like(params[k])
        for idx in np.ndindex(params[k].shape):
            old = params[k][idx]
            params[k][idx] = old + eps
            plus = loss_fn(params, forward(params, X)[0], y, l2)
            params[k][idx] = old - eps
            minus = loss_fn(params, forward(params, X)[0], y, l2)
            params[k][idx] = old
            numeric[idx] = (plus - minus) / (2 * eps)
        diff = np.linalg.norm(numeric - grads[k]) / (
            np.linalg.norm(numeric) + np.linalg.norm(grads[k]) + 1e-12)
        worst = max(worst, diff)
        print(f"  {k}: 相对误差 {diff:.2e}")
    print("梯度检查通过 ✓" if worst < 1e-6 else "梯度检查未通过，反向传播可能有误。")


# =====================================================================
# 7. 菜单
# =====================================================================
def ask_path():
    raw = input("请输入照片路径（可把照片直接拖进来）：").strip().strip("'\"")
    path = Path(raw.replace("\\ ", " ")).expanduser()
    if not path.is_file():
        print(f"找不到文件：{path}")
        return None
    return path


def run():
    DATA_DIR.mkdir(exist_ok=True)
    params = load_model()
    if params is None:
        print("还没有训练好的模型，现在开始第一次训练。")
        params = train()
    while True:
        print("\n【照片手写数字识别】")
        print("  1 —— 识别一张照片")
        print("  2 —— 重新训练模型")
        print("  3 —— 测试准确率（MNIST 测试集）")
        print("  4 —— 梯度检查")
        print("  0 —— 返回 / 退出")
        choice = input("请选择：").strip()
        if choice == "1":
            path = ask_path()
            if path:
                try:
                    recognize(params, path)
                except Exception as error:
                    print(f"识别失败：{error}")
        elif choice == "2":
            params = train()
        elif choice == "3":
            _, _, test_x, test_y = load_mnist()
            acc = accuracy(params, test_x.reshape(-1, 784) / 255.0, test_y)
            print(f"测试集准确率：{acc:.2%}（共 {len(test_y)} 张）")
        elif choice == "4":
            gradient_check()
        elif choice == "0":
            return
        else:
            print("输入无效。")


if __name__ == "__main__":
    run()
