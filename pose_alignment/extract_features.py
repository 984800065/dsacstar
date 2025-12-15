import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import faiss  # noqa: F401  # imported to ensure dependency is available when installing
import numpy as np
from PIL import Image
from loguru import logger
from tqdm.auto import tqdm

try:
    import torch
    import torchvision.transforms as T
except ImportError as e:  # pragma: no cover - 环境相关
    raise ImportError("本脚本需要 PyTorch 与 torchvision，请先安装相关依赖。") from e

try:
    import torch.hub as hub
except Exception:
    hub = None

# 确保可以作为普通脚本运行，不依赖包导入
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from utils_logging import setup_logger  # noqa: E402


def load_dinov2(device: str = "cpu") -> torch.nn.Module:
    """
    加载 DINOv2 模型。
    这里使用官方 torch hub / torch.hub 或其他公开权重加载方式。
    为保证工程可运行，你可以根据本地环境替换为合适的权重加载代码。
    """
    logger.info(f"正在加载 DINOv2 模型，device={device} ...")
    if hub is None:
        raise RuntimeError("torch.hub 不可用，请根据环境自行修改 DINOv2 加载逻辑。")

    # 示例：使用 Facebook 官方 dinov2-small 权重，可以根据需要替换
    model = hub.load("facebookresearch/dinov2", "dinov2_vits14")
    model.eval()
    model.to(device)
    logger.info("DINOv2 模型加载完成。")
    return model


def build_transform(image_size: int = 518) -> T.Compose:
    """
    输入图像预处理：
    - resize + center crop
    - 转 tensor
    - 使用 ImageNet 统计量做归一化（DINOv2 通常兼容该标准）
    """
    return T.Compose(
        [
            T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


def list_images(root: Path) -> List[Path]:
    rgb_dir = root / "train" / "rgb"
    exts = [".png", ".jpg", ".jpeg", ".bmp"]
    files = [p for p in sorted(rgb_dir.glob("*")) if p.suffix.lower() in exts]
    return files


@torch.no_grad()
def extract_features_for_dataset(
    dataset_root: Path,
    model: torch.nn.Module,
    device: str,
    batch_size: int = 16,
) -> Tuple[np.ndarray, List[str]]:
    """
    对一个数据集（4dclip1 或 renders）的 train/rgb 中所有图像提取全局特征。

    输出：
    - feats: (N, D) float32，L2 归一化后的特征矩阵
    - names: 长度 N 的列表，对应每个特征的图像基名（不含后缀）
    """
    transform = build_transform()
    images = list_images(dataset_root)
    logger.info(f"数据集 {dataset_root} 共找到 {len(images)} 张 rgb 图像。")

    feats: List[np.ndarray] = []
    names: List[str] = []

    batch_imgs: List[torch.Tensor] = []
    batch_names: List[str] = []

    for img_path in tqdm(images, desc=f"提取特征 - {dataset_root.name}"):
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.error(f"加载图像失败: {img_path}，错误：{e}")
            continue

        tensor = transform(img)
        batch_imgs.append(tensor)
        batch_names.append(img_path.stem)

        if len(batch_imgs) >= batch_size:
            batch_tensor = torch.stack(batch_imgs).to(device)
            emb = model(batch_tensor)  # 假设输出 (B, D)
            emb = emb.detach().cpu().numpy().astype("float32")
            # L2 归一化
            faiss.normalize_L2(emb)

            feats.append(emb)
            names.extend(batch_names)
            batch_imgs, batch_names = [], []

    # 处理最后一个 batch
    if batch_imgs:
        batch_tensor = torch.stack(batch_imgs).to(device)
        emb = model(batch_tensor)
        emb = emb.detach().cpu().numpy().astype("float32")
        faiss.normalize_L2(emb)
        feats.append(emb)
        names.extend(batch_names)

    if not feats:
        return np.zeros((0, 0), dtype="float32"), []

    feats_all = np.concatenate(feats, axis=0)
    logger.info(f"完成特征提取：{dataset_root.name}，形状={feats_all.shape}")
    return feats_all, names


def save_features(
    feats: np.ndarray,
    names: List[str],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        feats=feats,
        names=np.array(names),
    )
    logger.info(f"已保存特征到 {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="使用 DINOv2 提取两个数据集的图像特征")
    parser.add_argument(
        "--dataset1_root",
        type=str,
        default="4dclip1",
        help="Dataset 1 相对当前脚本目录的根目录",
    )
    parser.add_argument(
        "--dataset2_root",
        type=str,
        default="renders",
        help="Dataset 2 相对当前脚本目录的根目录",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="features",
        help="特征缓存输出目录（相对当前脚本目录）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="计算设备，如 cuda 或 cpu",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="特征提取 batch size",
    )

    args = parser.parse_args()
    setup_logger()
    logger.info(
        f"启动特征提取脚本，"
        f"dataset1_root={args.dataset1_root}, "
        f"dataset2_root={args.dataset2_root}, "
        f"output_dir={args.output_dir}, "
        f"device={args.device}, "
        f"batch_size={args.batch_size}"
    )

    device = args.device
    model = load_dinov2(device=device)

    # 所有路径都基于脚本所在目录，避免依赖运行时工作目录
    dataset1_root = CURRENT_DIR / args.dataset1_root
    dataset2_root = CURRENT_DIR / args.dataset2_root
    output_dir = CURRENT_DIR / args.output_dir

    feats1, names1 = extract_features_for_dataset(
        dataset_root=dataset1_root,
        model=model,
        device=device,
        batch_size=args.batch_size,
    )
    feats2, names2 = extract_features_for_dataset(
        dataset_root=dataset2_root,
        model=model,
        device=device,
        batch_size=args.batch_size,
    )

    save_features(
        feats1,
        names1,
        output_dir / "4dclip1_features.npz",
    )
    save_features(
        feats2,
        names2,
        output_dir / "renders_features.npz",
    )


if __name__ == "__main__":
    main()


