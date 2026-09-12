"""沙箱子进程入口：先装围栏，再跑运行目录里的 script.py。

父进程（geo_compute.sandbox）负责超时、内存与进程数上限；本文件负责子进程内部的
禁网、导入黑名单与文件围栏，并把 stdout、异常与结果写成 _result.json。
护栏与用户代码同在内核里，拦的是常规写法——防事故，不防攻击（L2 交给 Docker）。
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import traceback
from pathlib import Path

JOB_FILE = "_job.json"
RESULT_FILE = "_result.json"
SCRIPT_FILE = "script.py"
TEXT_LIMIT = 8000

# 读放行：运行目录、data/，以及解释器与缓存（导入系统与字体缓存要用）
READABLE_SUBDIRS = ("data", ".venv", ".uv-cache")


def _install_network_guard() -> None:
    import socket

    def blocked(*args, **kwargs):
        raise PermissionError("沙箱禁网：本次执行不允许访问网络")

    socket.socket = blocked
    socket.create_connection = blocked
    socket.getaddrinfo = blocked
    socket.gethostbyname = blocked


def _install_import_guard(deny: tuple[str, ...]) -> None:
    import builtins

    real_import = builtins.__import__

    def guarded(name, globals=None, locals=None, fromlist=(), level=0):
        if level == 0 and name.split(".")[0] in deny:
            raise ImportError(f"沙箱禁用模块 {name}：本次执行不允许联网或起进程")
        return real_import(name, globals, locals, fromlist, level)

    builtins.__import__ = guarded


def _install_fs_guard(run_dir: Path, root: Path) -> None:
    import builtins

    real_open = io.open
    readable_extra = [root / sub for sub in READABLE_SUBDIRS]

    def readable(path: Path) -> bool:
        if path.is_relative_to(run_dir) or not path.is_relative_to(root):
            return True
        return any(path.is_relative_to(sub) for sub in readable_extra)

    def guarded_open(file, mode="r", *args, **kwargs):
        if isinstance(file, int):
            return real_open(file, mode, *args, **kwargs)
        path = Path(os.fspath(file)).resolve()
        writing = any(flag in mode for flag in "wax+")
        if writing and not path.is_relative_to(run_dir):
            raise PermissionError(f"沙箱只允许写运行目录，拒绝写 {path}")
        if not writing and not readable(path):
            raise PermissionError(f"沙箱拒绝读取项目内文件 {path}：只放行 data/ 与运行目录")
        return real_open(file, mode, *args, **kwargs)

    io.open = guarded_open
    builtins.open = guarded_open

    def guarded_os_open(path, flags, *args, **kwargs):
        path = Path(os.fspath(path)).resolve()
        writing = bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
        if writing and not path.is_relative_to(run_dir):
            raise PermissionError(f"沙箱只允许写运行目录，拒绝写 {path}")
        if not writing and not readable(path):
            raise PermissionError(f"沙箱拒绝读取项目内文件 {path}：只放行 data/ 与运行目录")
        return real_os_open(path, flags, *args, **kwargs)

    real_os_open = os.open
    os.open = guarded_os_open

    def scoped(name: str, arity: int):
        real = getattr(os, name, None)
        if real is None:
            return

        def guarded(*args, **kwargs):
            for raw in args[:arity]:
                if isinstance(raw, (str, bytes, os.PathLike)):
                    path = Path(os.fspath(raw)).resolve()
                    if not path.is_relative_to(run_dir):
                        raise PermissionError(f"沙箱拒绝 {name} 作用于 {path}：只许在运行目录内")
            return real(*args, **kwargs)

        setattr(os, name, guarded)

    for name, arity in (("remove", 1), ("unlink", 1), ("rmdir", 1), ("removedirs", 1),
                        ("mkdir", 1), ("makedirs", 1), ("chmod", 1), ("truncate", 1),
                        ("utime", 1), ("symlink", 1), ("link", 1), ("rename", 2),
                        ("replace", 2)):
        scoped(name, arity)

    def blocked_call(name: str):
        def guard(*args, **kwargs):
            raise PermissionError(f"沙箱禁止 {name}：本次执行不允许起子进程")

        return guard

    for name in ("system", "popen", "startfile", "fork", "kill", "killpg", "abort"):
        if hasattr(os, name):
            setattr(os, name, blocked_call(f"os.{name}"))
    for name in dir(os):
        if name.startswith("exec") or name.startswith("spawn"):
            setattr(os, name, blocked_call(f"os.{name}"))


def _build_namespace(run_dir: Path, root: Path) -> dict:
    """预置常用库与只读 DuckDB 连接，让模型专心写分析而不是写样板。"""
    import duckdb
    import geopandas as gpd
    import numpy as np
    import pandas as pd
    import pyproj
    import shapely

    from geo_compute import query

    db = root / "data" / "processed" / "geo.duckdb"
    con = duckdb.connect(str(db), read_only=True)
    return {
        "__name__": "__sandbox__",
        "__file__": str(run_dir / SCRIPT_FILE),
        "con": con,
        "query": query,
        "duckdb": duckdb,
        "pd": pd,
        "np": np,
        "gpd": gpd,
        "shapely": shapely,
        "pyproj": pyproj,
        "RUN_DIR": run_dir,
        "OUT_DIR": run_dir,
        "TMP_DIR": run_dir / "tmp",
        "DATA_DIR": root / "data",
        "DB": str(db),
    }


def _clean_traceback(text: str) -> str:
    """只留 script.py 之后的帧——沙箱自己的栈对写代码的人没有意义。"""
    lines = text.strip().splitlines()
    for index, line in enumerate(lines):
        if f'File "{SCRIPT_FILE}"' in line:
            return "\n".join(lines[index - 1:]) if index else "\n".join(lines)
    return text.strip()


def _jsonable(value, depth: int = 0):
    if depth > 3:
        return str(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v, depth + 1) for k, v in list(value.items())[:50]}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v, depth + 1) for v in list(value)[:50]]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            return _jsonable(to_dict(orient="records"), depth + 1)
        except TypeError:
            return _jsonable(to_dict(), depth + 1)
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return value.item()
        except Exception:
            pass
    wkt = getattr(value, "wkt", None)
    if isinstance(wkt, str):
        return wkt
    return str(value)


def _clip(text: str) -> str:
    text = text or ""
    return text if len(text) <= TEXT_LIMIT else text[:TEXT_LIMIT] + "\n…（已截断）"


def main() -> int:
    run_dir = Path(sys.argv[1]).resolve()
    root = run_dir.parents[1]
    job = json.loads((run_dir / JOB_FILE).read_text(encoding="utf-8"))

    namespace = _build_namespace(run_dir, root)
    _install_network_guard()
    _install_fs_guard(run_dir, root)
    _install_import_guard(tuple(job.get("deny_modules", ())))

    code = (run_dir / SCRIPT_FILE).read_text(encoding="utf-8")
    payload = {"ok": False, "stdout": "", "stderr": "", "error": None,
               "error_type": None, "traceback": None, "result": None}
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            exec(compile(code, SCRIPT_FILE, "exec"), namespace)
            payload["ok"] = True
        except BaseException as exc:  # noqa: BLE001 —— 沙箱要吞掉一切并如实回报
            payload["error"] = f"{type(exc).__name__}: {exc}"
            payload["error_type"] = type(exc).__name__
            payload["traceback"] = _clean_traceback(traceback.format_exc())

    payload["stdout"] = _clip(out.getvalue())
    payload["stderr"] = _clip(err.getvalue())
    payload["result"] = _jsonable(namespace.get("RESULT", namespace.get("result")))
    (run_dir / RESULT_FILE).write_text(
        json.dumps(payload, ensure_ascii=False, indent=1, default=str),
        encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())