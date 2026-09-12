"""受限代码执行（L1：本地受限子进程）。

口径见 AGENTS.md 硬性规则与 HANDOFF.md 3.5：Agent 生成的代码一律在 sandbox/run_<id>/
内执行，禁网、有超时与内存上限、数据只读，执行记录（代码 / 输出 / 产物哈希）可回放。

执行器是可替换后端：L1 在本机跑受限子进程，将来换 Docker（L2）只换这一个类。
L1 的三道闸：
  1. 父进程——Windows Job Object 压住内存与进程数上限，超时则整棵进程树终止
  2. 子进程——导入黑名单 + socket 守卫 + open/os 路径围栏（只许写运行目录）
  3. 事前——AST 静态检查，把 os.system / subprocess / shutil.rmtree 这类调用挡在执行前

已知局限（L1 接受、L2 解决）：护栏与用户代码在同一个内核里，拦的是常规写法而不是
内核隔离；子进程加入 Job Object 之前有一个微秒级窗口。所以 L1 防的是事故，不是攻击。
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from geo_compute import query

ROOT = query.ROOT
SANDBOX_ROOT = ROOT / "sandbox"
RUNNER_MODULE = "geo_compute.sandbox_runner"

DEFAULT_TIMEOUT_S = 30.0
MAX_TIMEOUT_S = 120.0
DEFAULT_MEMORY_MB = 1024
MAX_MEMORY_MB = 4096
MAX_PROCESSES = 1
OUTPUT_LIMIT = 8000
ARTIFACT_LIMIT = 20


def boot_processes() -> int:
    """启动链要放行几个进程。

    uv 建的 .venv\\Scripts\\python.exe 是 trampoline——它自己会再起一层真解释器，
    所以「进程数上限 = 1」会把启动链一起掐死。这里的上限是第二道闸，第一道是
    子进程里的导入黑名单与 os.* 围栏；用户代码想再起进程，两道都过不去。
    """
    base = getattr(sys, "_base_executable", sys.executable)
    return 2 if base != sys.executable else 1

# 子进程里禁止 import 的顶层模块：能开网络、能起进程、能碰内核的都不给
DENY_MODULES = (
    "subprocess", "socket", "ssl", "ctypes", "cffi", "multiprocessing", "concurrent",
    "asyncio", "urllib", "http", "ftplib", "smtplib", "poplib", "imaplib", "telnetlib",
    "xmlrpc", "requests", "httpx", "aiohttp", "urllib3", "webbrowser", "winreg",
    "pty", "tty", "fcntl", "resource", "win32api", "win32com", "win32con",
)

# 明确的危险调用：即便调用方绕过了 import 检查，也在执行前拦下
DENY_CALLS = {
    "os.system", "os.popen", "os.startfile", "os.fork", "os.abort", "os.kill", "os.killpg",
    "shutil.rmtree", "subprocess.run", "subprocess.Popen", "subprocess.call",
    "subprocess.check_output", "subprocess.check_call",
}
DENY_CALL_PREFIXES = ("os.exec", "os.spawn", "subprocess.")


def _dotted(node: ast.AST) -> str:
    """把 a.b.c 形式的表达式还原成点号字符串；不是纯名字/属性链就返回空串。"""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return ""


def check_code(code: str) -> list[str]:
    """AST 静态检查：只看导入与调用，不执行代码。返回问题列表，空表示通过。"""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [f"语法错误：第 {exc.lineno} 行 {exc.msg}"]
    findings: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in DENY_MODULES:
                    findings.append(f"第 {node.lineno} 行：禁止导入 {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] in DENY_MODULES:
                findings.append(f"第 {node.lineno} 行：禁止从 {node.module} 导入")
        elif isinstance(node, ast.Call):
            name = _dotted(node.func)
            if name in DENY_CALLS or name.startswith(DENY_CALL_PREFIXES):
                findings.append(f"第 {node.lineno} 行：禁止调用 {name}")
    return findings


@dataclass
class ExecResult:
    ok: bool
    run_id: str
    run_dir: str
    code: str
    purpose: str = ""
    stdout: str = ""
    stderr: str = ""
    error: str | None = None
    error_type: str | None = None
    traceback_text: str | None = None
    result: Any = None
    artifacts: list[dict] = field(default_factory=list)
    elapsed_s: float = 0.0
    timed_out: bool = False
    killed_reason: str | None = None
    static_findings: list[str] = field(default_factory=list)
    limits: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "run_id": self.run_id,
            "run_dir": self.run_dir,
            "purpose": self.purpose,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "error": self.error,
            "error_type": self.error_type,
            "traceback": self.traceback_text,
            "result": self.result,
            "artifacts": self.artifacts,
            "elapsed_s": self.elapsed_s,
            "timed_out": self.timed_out,
            "killed_reason": self.killed_reason,
            "static_findings": self.static_findings,
            "limits": self.limits,
        }


class Sandbox(Protocol):
    def run(self, code: str, *, purpose: str = "", timeout_s: float = DEFAULT_TIMEOUT_S,
            memory_mb: int = DEFAULT_MEMORY_MB) -> ExecResult: ...


class LocalProcessSandbox:
    """L1 后端：本机受限子进程。"""

    def __init__(self, root: Path = SANDBOX_ROOT):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def run(self, code: str, *, purpose: str = "", timeout_s: float = DEFAULT_TIMEOUT_S,
            memory_mb: int = DEFAULT_MEMORY_MB) -> ExecResult:
        timeout_s = float(min(max(timeout_s, 1.0), MAX_TIMEOUT_S))
        memory_mb = int(min(max(memory_mb, 128), MAX_MEMORY_MB))
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"{stamp}_{uuid.uuid4().hex[:8]}"
        run_dir = self.root / f"run_{run_id}"
        (run_dir / "tmp").mkdir(parents=True, exist_ok=True)
        (run_dir / "script.py").write_text(code, encoding="utf-8", newline="\n")
        job = {
            "run_id": run_id,
            "purpose": purpose,
            "timeout_s": timeout_s,
            "memory_mb": memory_mb,
            "max_processes": MAX_PROCESSES,
            "deny_modules": list(DENY_MODULES),
        }
        (run_dir / "_job.json").write_text(json.dumps(job, ensure_ascii=False, indent=1),
                                           encoding="utf-8", newline="\n")

        max_processes = boot_processes()
        limits = {"timeout_s": timeout_s, "memory_mb": memory_mb,
                  "max_processes": max_processes, "backend": type(self).__name__}
        findings = check_code(code)
        if findings:
            return ExecResult(ok=False, run_id=run_id, run_dir=str(run_dir), code=code,
                              purpose=purpose, error="静态检查未通过：" + "；".join(findings),
                              error_type="StaticCheckError", static_findings=findings,
                              limits=limits)

        started = time.perf_counter()
        env = _child_env(run_dir)
        kwargs: dict[str, Any] = {}
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        else:
            kwargs["preexec_fn"] = _posix_limits(memory_mb, max_processes)

        proc = subprocess.Popen(
            [sys.executable, "-P", "-m", RUNNER_MODULE, str(run_dir)],
            cwd=str(run_dir), env=env, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kwargs,
        )
        if os.name == "nt":
            job_handle, job_note = _assign_job(proc, memory_mb, max_processes)
        else:
            job_handle, job_note = None, ""
        if job_note:
            limits["job_note"] = job_note

        stdout = stderr = ""
        timed_out = False
        killed_reason = None
        try:
            out_b, err_b = proc.communicate(timeout=timeout_s)
            stdout, stderr = _decode(out_b), _decode(err_b)
        except subprocess.TimeoutExpired:
            timed_out = True
            killed_reason = f"超过 {timeout_s:g} 秒，已终止整棵进程树"
            _terminate_job(job_handle, proc)
            out_b, err_b = proc.communicate()
            stdout, stderr = _decode(out_b), _decode(err_b)
        finally:
            _close_job(job_handle)
        elapsed = round(time.perf_counter() - started, 3)

        payload = _read_result(run_dir)
        artifacts = _collect_artifacts(run_dir)
        if payload is None and not timed_out:
            hint = ""
            if "Memory allocation" in stderr or "MemoryError" in stderr:
                hint = f"——看起来是内存上限 {memory_mb} MB 生效，子进程被分配失败拖死"
            error = (f"沙箱子进程未产出结果（退出码 {proc.returncode}）{hint}；"
                     f"stderr：{stderr.strip()[:400] or '（空）'}")
            return ExecResult(ok=False, run_id=run_id, run_dir=str(run_dir), code=code,
                              purpose=purpose, stdout=_clip(stdout), stderr=_clip(stderr),
                              error=error, error_type="SandboxCrashed",
                              artifacts=artifacts, elapsed_s=elapsed,
                              killed_reason=killed_reason, static_findings=[],
                              limits=limits)

        payload = payload or {}
        return ExecResult(
            ok=bool(payload.get("ok")) and not timed_out,
            run_id=run_id, run_dir=str(run_dir), code=code, purpose=purpose,
            stdout=_clip(payload.get("stdout", stdout)), stderr=_clip(payload.get("stderr", stderr)),
            error=killed_reason if timed_out else payload.get("error"),
            error_type="Timeout" if timed_out else payload.get("error_type"),
            traceback_text=payload.get("traceback"), result=payload.get("result"),
            artifacts=artifacts, elapsed_s=elapsed, timed_out=timed_out,
            killed_reason=killed_reason, static_findings=[], limits=limits,
        )


def _child_env(run_dir: Path) -> dict[str, str]:
    """删掉一切看起来像凭据的环境变量——沙箱里不该有 .env 的内容。"""
    secretish = ("KEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL", "AUTH")
    env = {k: v for k, v in os.environ.items()
           if not any(word in k.upper() for word in secretish)}
    env.update({
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUTF8": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "TEMP": str(run_dir / "tmp"),
        "TMP": str(run_dir / "tmp"),
        "MPLCONFIGDIR": str(run_dir / "tmp"),
    })
    return env


def _decode(raw: bytes | None) -> str:
    return (raw or b"").decode("utf-8", errors="replace")


def _clip(text: str) -> str:
    text = text or ""
    if len(text) <= OUTPUT_LIMIT:
        return text
    return text[:OUTPUT_LIMIT] + f"\n…（输出超长，已截断，完整输出见运行目录）"


def _read_result(run_dir: Path) -> dict | None:
    path = run_dir / "_result.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _collect_artifacts(run_dir: Path) -> list[dict]:
    skip = {"script.py", "_job.json", "_result.json"}
    out: list[dict] = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path.name in skip or path.parent.name == "tmp":
            continue
        data = path.read_bytes()
        out.append({"path": path.relative_to(run_dir).as_posix(), "bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest()[:16]})
        if len(out) >= ARTIFACT_LIMIT:
            break
    return out


def _assign_job(proc: subprocess.Popen, memory_mb: int, max_processes: int):
    """Windows：给子进程套一个 Job Object，压住内存与进程数。

    套不上就如实记一笔再降级——超时兜底仍在，内存上限会失效，这不能假装成功。
    """
    try:
        from geo_compute.winjob import WindowsJob

        job = WindowsJob(memory_mb=memory_mb, max_processes=max_processes)
        job.assign(proc._handle)
        return job, ""
    except Exception as exc:
        return None, f"Job Object 未生效（{type(exc).__name__}: {exc}），本次只有超时兜底"


def _terminate_job(job, proc: subprocess.Popen) -> None:
    try:
        if job is not None:
            job.terminate()
        else:
            proc.kill()
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def _close_job(job) -> None:
    if job is not None:
        try:
            job.close()
        except Exception:
            pass


def _posix_limits(memory_mb: int, max_processes: int):
    """POSIX 侧的等价护栏（当前开发机是 Windows，这里是给换机留的口子）。"""
    if os.name == "nt":
        return None

    def apply() -> None:
        import resource

        limit = memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
        resource.setrlimit(resource.RLIMIT_NPROC, (max_processes, max_processes))

    return apply
