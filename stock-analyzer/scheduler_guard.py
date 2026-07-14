"""调度与推送防重 — 单实例锁、同槽位去重。"""

from __future__ import annotations

import fcntl
import json
import os
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator

import config
from utils import SetupLogger

logger = SetupLogger(config.LOG_LEVEL)

_GUARD_DIR = Path(__file__).resolve().parent
_SCHEDULER_LOCK_PATH = _GUARD_DIR / ".scheduler.lock"
_PUSH_DEDUP_PATH = _GUARD_DIR / ".push_dedup.json"
_DEFAULT_PUSH_TTL_MINUTES = 8


@contextmanager
def _OpenLockFile(path: Path) -> Iterator[int]:
    """打开锁文件并返回文件描述符。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        yield fd
    finally:
        os.close(fd)


def TryAcquireSchedulerLock() -> bool:
    """尝试获取定时调度单实例锁；已有实例在跑时返回 False。"""
    try:
        fd = os.open(str(_SCHEDULER_LOCK_PATH), os.O_CREAT | os.O_RDWR, 0o644)
    except OSError as exc:
        logger.error("无法创建调度锁文件: %s", exc)
        return False
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        logger.error(
            "已有 main.py 定时调度在运行，请勿重复启动。"
            "可用 ps aux | grep main.py 检查并结束多余进程。"
        )
        return False
    except OSError as exc:
        os.close(fd)
        logger.error("获取调度锁失败: %s", exc)
        return False

    try:
        os.ftruncate(fd, 0)
        os.write(fd, f"{os.getpid()}\n".encode())
    except OSError as exc:
        logger.warning("写入调度锁 PID 失败: %s", exc)
    return True


def _LoadPushDedupState() -> dict[str, str]:
    """读取推送去重状态。"""
    if not _PUSH_DEDUP_PATH.exists():
        return {}
    try:
        raw = _PUSH_DEDUP_PATH.read_text(encoding="utf-8").strip()
        if not raw:
            return {}
        data = json.loads(raw)
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        logger.warning("读取推送去重状态失败: %s", exc)
    return {}


def _SavePushDedupState(state: dict[str, str]) -> None:
    """保存推送去重状态。"""
    try:
        _PUSH_DEDUP_PATH.write_text(
            json.dumps(state, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError as exc:
        logger.warning("写入推送去重状态失败: %s", exc)


def _PruneExpiredSlots(
    state: dict[str, str],
    ttl_minutes: int,
) -> dict[str, str]:
    """剔除已过期的推送槽位记录。"""
    cutoff = datetime.now() - timedelta(minutes=ttl_minutes)
    kept: dict[str, str] = {}
    for key, ts_text in state.items():
        try:
            ts = datetime.fromisoformat(ts_text)
        except ValueError:
            continue
        if ts >= cutoff:
            kept[key] = ts_text
    return kept


def BuildPushSlotKey(
    push_time: str | None,
    session_label: str,
    stock_code: str | None = None,
) -> str:
    """构建当日推送槽位键（用于去重）。"""
    day = datetime.now().strftime("%Y-%m-%d")
    slot = push_time.strip() if push_time and push_time.strip() else session_label.strip()
    code = (stock_code or config.STOCK_CODE).strip()
    return f"{day}:{slot}:{code}"


def TryClaimPushSlot(
    push_time: str | None,
    session_label: str,
    ttl_minutes: int = _DEFAULT_PUSH_TTL_MINUTES,
    stock_code: str | None = None,
) -> bool:
    """
    尝试认领推送槽位。
    同一自然日、同一 push_time/时段 在 ttl 内仅允许推送一次。
    """
    slot_key = BuildPushSlotKey(push_time, session_label, stock_code)
    try:
        with _OpenLockFile(_PUSH_DEDUP_PATH) as fd:
            fcntl.flock(fd, fcntl.LOCK_EX)
            state = _PruneExpiredSlots(_LoadPushDedupState(), ttl_minutes)
            if slot_key in state:
                logger.warning(
                    "跳过重复推送 — 槽位 %s 已在 %s 推送过（%d 分钟内不重复）",
                    slot_key,
                    state[slot_key],
                    ttl_minutes,
                )
                return False
            state[slot_key] = datetime.now().isoformat(timespec="seconds")
            _SavePushDedupState(state)
            return True
    except OSError as exc:
        logger.error("推送去重锁异常，为安全起见不推送: %s", exc)
        return False


def ReleasePushSlot(
    push_time: str | None,
    session_label: str,
    stock_code: str | None = None,
) -> None:
    """推送失败时释放槽位，便于重试。"""
    slot_key = BuildPushSlotKey(push_time, session_label, stock_code)
    try:
        with _OpenLockFile(_PUSH_DEDUP_PATH) as fd:
            fcntl.flock(fd, fcntl.LOCK_EX)
            state = _LoadPushDedupState()
            if slot_key in state:
                del state[slot_key]
                _SavePushDedupState(state)
    except OSError as exc:
        logger.warning("释放推送槽位失败: %s", exc)
