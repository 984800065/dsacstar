import argparse
import csv
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np
from loguru import logger
from tqdm.auto import tqdm

# 确保可以作为普通脚本运行，不依赖包导入
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from utils_logging import setup_logger  # noqa: E402


def load_features(
    feat_path: Path,
) -> Tuple[np.ndarray, List[str]]:
    data = np.load(feat_path)
    feats = data["feats"].astype("float32")
    names = data["names"].tolist()
    return feats, names


def build_faiss_index(feats: np.ndarray) -> faiss.Index:
    """
    使用 FAISS 构建余弦相似度索引。
    由于特征已经做过 L2 归一化，内积即为余弦相似度。
    """
    if feats.ndim != 2:
        raise ValueError(f"特征矩阵维度不正确: {feats.shape}")
    dim = feats.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(feats)
    logger.info(f"FAISS 索引构建完成，特征数={feats.shape[0]}，维度={dim}")
    return index


def one_to_one_top1_matching(
    feats1: np.ndarray,
    names1: List[str],
    feats2: np.ndarray,
    names2: List[str],
    index2: faiss.Index,
    similarity_threshold: float,
    similarity_upper_threshold: float,
) -> List[Tuple[str, str, float]]:
    """
    实现 Dataset1 -> Dataset2 的一对一 Top-1 匹配 + 阈值 cut。

    返回：
    - (name1, name2, score) 列表，仅包含匹配成功样本。
    """
    if feats1.shape[0] == 0 or feats2.shape[0] == 0:
        logger.warning("任一数据集特征为空，无法进行匹配。")
        return []

    n1 = feats1.shape[0]
    used_mask = np.zeros(feats2.shape[0], dtype=bool)
    idx_name2: Dict[int, str] = {i: n for i, n in enumerate(names2)}

    results: List[Tuple[str, str, float]] = []

    # 按照要求使用 tqdm 显示进度
    for i in tqdm(range(n1), desc="执行 Top-1 匹配"):
        query = feats1[i : i + 1]  # (1, D)
        # 检索 Top-K，这里先取 K= min(10, N2) 以便在已用掉的候选中继续向后找
        k = min(10, feats2.shape[0])
        scores, idxs = index2.search(query, k)
        scores = scores[0]
        idxs = idxs[0]

        matched_name2 = None
        matched_score = None

        for score, idx in zip(scores, idxs):
            if idx < 0:
                continue
            if used_mask[idx]:
                continue
            matched_name2 = idx_name2[idx]
            matched_score = float(score)
            break

        if matched_name2 is None:
            # 没有找到未使用的候选
            continue

        # 下限阈值 cut：相似度太低直接丢弃
        if matched_score < similarity_threshold:
            continue

        # 上限阈值 cut：相似度高于给定上界也丢弃（用于过滤过于相似/潜在重复）
        if matched_score > similarity_upper_threshold:
            continue

        used_mask[idxs[0]] = True
        results.append((names1[i], matched_name2, matched_score))

        if len(results) % 50 == 0:
            logger.info(f"当前已匹配成功 {len(results)} 对样本。")

    logger.info(f"最终匹配成功样本数：{len(results)}")
    return results


def copy_and_rename_pairs(
    dataset1_root: Path,
    dataset2_root: Path,
    matches: List[Tuple[str, str, float]],
    output_root: Path,
) -> None:
    """
    仅对匹配成功的样本进行文件复制与重命名。

    新目录结构：
    result/
      sub_4dclip1/train/{rgb, calibration, poses}
      sub_renders/train/{rgb, calibration, poses}
    文件名统一从 0 开始编号。
    """
    sub1_root = output_root / "sub_4dclip1" / "train"
    sub2_root = output_root / "sub_renders" / "train"

    for sub_root in [sub1_root, sub2_root]:
        for sub_dir in ["rgb", "calibration", "poses"]:
            (sub_root / sub_dir).mkdir(parents=True, exist_ok=True)

    # 同时在这里写出 matches.csv，保证 new_id 与拷贝顺序一致
    csv_path = output_root / "matches.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_file = csv_path.open("w", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    writer.writerow(["new_id", "name_4dclip1", "name_renders", "similarity"])

    def copy_triplet(
        src_root: Path,
        basename: str,
        dst_root: Path,
        new_id: int,
    ) -> None:
        rgb_src = src_root / "train" / "rgb"
        cali_src = src_root / "train" / "calibration"
        poses_src = src_root / "train" / "poses"

        # 保持原扩展名
        rgb_path = next((p for p in rgb_src.glob(f"{basename}.*")), None)
        cali_path = next((p for p in cali_src.glob(f"{basename}.*")), None)
        pose_path = next((p for p in poses_src.glob(f"{basename}.*")), None)

        if rgb_path is None or cali_path is None or pose_path is None:
            logger.warning(f"缺失文件，跳过样本 {basename}: {rgb_path}, {cali_path}, {pose_path}")
            return

        new_rgb = dst_root / "rgb" / f"{new_id}{rgb_path.suffix}"
        new_cali = dst_root / "calibration" / f"{new_id}{cali_path.suffix}"
        new_pose = dst_root / "poses" / f"{new_id}{pose_path.suffix}"

        shutil.copy2(rgb_path, new_rgb)
        shutil.copy2(cali_path, new_cali)
        shutil.copy2(pose_path, new_pose)

    for new_id, (name1, name2, score) in enumerate(
        tqdm(matches, desc="复制并重命名匹配样本")
    ):
        logger.debug(f"复制样本对 {new_id}: {name1} <-> {name2} (score={score:.4f})")
        # 写入匹配元数据
        writer.writerow([new_id, name1, name2, f"{score:.6f}"])
        # 复制对应三元文件
        copy_triplet(dataset1_root, name1, sub1_root, new_id)
        copy_triplet(dataset2_root, name2, sub2_root, new_id)

    csv_file.close()
    logger.info(f"结果文件与 matches.csv 写入完成，输出目录：{output_root}")


def main() -> None:
    parser = argparse.ArgumentParser(description="利用 DINOv2 特征与 FAISS 进行图像匹配")
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
        "--features_dir",
        type=str,
        default="features",
        help="特征缓存目录（相对当前脚本目录）",
    )
    parser.add_argument(
        "--similarity_threshold",
        type=float,
        default=0.0,
        help="相似度阈值（余弦相似度），小于该值则不认为匹配成功",
    )
    parser.add_argument(
        "--similarity_upper_threshold",
        type=float,
        default=1.0,
        help="相似度上限阈值，大于该值也不认为匹配成功（用于过滤过于相似的样本），默认 1.0 表示不启用",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="result",
        help="结果输出根目录（相对当前脚本目录）",
    )

    args = parser.parse_args()
    setup_logger(log_name="match.log")

    logger.info(
        "启动匹配脚本，参数："
        f"dataset1_root={args.dataset1_root}, "
        f"dataset2_root={args.dataset2_root}, "
        f"features_dir={args.features_dir}, "
        f"similarity_threshold={args.similarity_threshold}, "
        f"similarity_upper_threshold={args.similarity_upper_threshold}, "
        f"output_root={args.output_root}"
    )

    # 所有路径都基于脚本所在目录，避免依赖运行时工作目录
    features_dir = CURRENT_DIR / args.features_dir
    feat1_path = features_dir / "4dclip1_features.npz"
    feat2_path = features_dir / "renders_features.npz"

    feats1, names1 = load_features(feat1_path)
    feats2, names2 = load_features(feat2_path)
    logger.info(
        f"加载特征完成：4dclip1 N={feats1.shape[0]}，renders N={feats2.shape[0]}"
    )

    index2 = build_faiss_index(feats2)

    matches = one_to_one_top1_matching(
        feats1=feats1,
        names1=names1,
        feats2=feats2,
        names2=names2,
        index2=index2,
        similarity_threshold=args.similarity_threshold,
        similarity_upper_threshold=args.similarity_upper_threshold,
    )

    dataset1_root = CURRENT_DIR / args.dataset1_root
    dataset2_root = CURRENT_DIR / args.dataset2_root
    output_root = CURRENT_DIR / args.output_root

    copy_and_rename_pairs(
        dataset1_root=dataset1_root,
        dataset2_root=dataset2_root,
        matches=matches,
        output_root=output_root,
    )


if __name__ == "__main__":
    main()


