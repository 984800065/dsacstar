from pathlib import Path
import sys

from loguru import logger


def setup_logger(log_dir: str = "pose_alignment/logs", log_name: str = "match.log") -> None:
    """
    统一配置 loguru 日志:
    - 同时输出到控制台与文件
    - 日志格式包含时间、等级、文件路径、行号、消息
    """
    # 移除默认 handler，避免重复输出
    logger.remove()

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    log_file = log_path / log_name

    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "{file.path}:{line} | "
        "<level>{message}</level>"
    )

    # 控制台输出（使用标准流 + colorize，保证有颜色）
    logger.add(
        sys.stderr,
        format=log_format,
        level="INFO",
        colorize=True,
        enqueue=True,
        backtrace=True,
        diagnose=False,
    )

    # 文件输出
    logger.add(
        log_file,
        rotation="10 MB",
        retention=5,
        encoding="utf-8",
        format=log_format,
        level="INFO",
        enqueue=True,
        backtrace=True,
        diagnose=False,
    )


__all__ = ["logger", "setup_logger"]


