"""沙箱验收：逐条验证 L1 的三道闸与产物回放。

每次运行都要真的起子进程、真的触发超时与内存上限，因此比普通单测慢（约 20 秒）。
用法：uv run python scripts/sandbox_smoke_test.py
"""

from __future__ import annotations

import sys

from geo_compute.sandbox import LocalProcessSandbox

CASES: list[tuple[str, str, dict]] = [
    (
        "正常执行：查库 + 打印 + 落产物 + 回传 RESULT",
        """
rows = con.execute("select district, count(*) c from poi_point "
                   "where district is not null group by 1 order by 2 desc").fetchall()
print("面点层按区计数：", rows)
(RUN_DIR / "counts.csv").write_text(
    "\\n".join(f"{name},{count}" for name, count in rows), encoding="utf-8")
RESULT = {"districts": len(rows), "top": rows[0][0]}
""",
        {"ok": True, "expect_stdout": "面点层按区计数", "expect_artifacts": ["counts.csv"]},
    ),
    (
        "静态检查：import subprocess 在执行前就被拦下",
        "import subprocess\nsubprocess.run(['cmd', '/c', 'echo hi'])\n",
        {"ok": False, "expect_error_type": "StaticCheckError"},
    ),
    (
        "静态检查：shutil.rmtree 被拦下",
        "import shutil\nshutil.rmtree('D:/')\n",
        {"ok": False, "expect_error_type": "StaticCheckError"},
    ),
    (
        "禁网：绕过静态检查后，运行时导入守卫依然拦住 socket",
        "socket = __import__('socket')\nsocket.socket()\n",
        {"ok": False, "expect_error_type": "ImportError"},
    ),
    (
        "禁网：urllib.request 同样进不来",
        "__import__('urllib.request')\n",
        {"ok": False, "expect_error_type": "ImportError"},
    ),
    (
        "禁起进程：绕过静态检查拿到 os.system，运行时围栏依然拒绝",
        "import os\nsystem = getattr(os, 'system')\nsystem('echo hi')\n",
        {"ok": False, "expect_error_type": "PermissionError"},
    ),
    (
        "文件围栏：写运行目录之外被拒绝",
        "open('D:/AAAAAAAAAAAAARJGC/YOURWORLD/leak.txt', 'w').write('x')\n",
        {"ok": False, "expect_error_type": "PermissionError"},
    ),
    (
        "文件围栏：读项目内的 .env 被拒绝",
        "print(open('D:/AAAAAAAAAAAAARJGC/YOURWORLD/.env').read())\n",
        {"ok": False, "expect_error_type": "PermissionError"},
    ),
    (
        "超时：死循环在 4 秒后被整棵进程树终止",
        "while True:\n    pass\n",
        {"ok": False, "expect_timeout": True, "timeout_s": 4.0},
    ),
    (
        "内存上限：256 MB 上限下分配 1 GB 会失败而不是拖垮机器",
        "block = []\nfor _ in range(64):\n    block.append(bytearray(16 * 1024 * 1024))\nprint('分配完成', len(block))\n",
        {"ok": False, "memory_mb": 256, "expect_error_any": True,
         "expect_stdout_absent": "分配完成"},
    ),
    (
        "异常如实回报：traceback 指向 script.py 而不是沙箱自己",
        "values = [1, 2]\nprint(values[9])\n",
        {"ok": False, "expect_error_type": "IndexError", "expect_trace_mentions": "script.py"},
    ),
]


def main() -> int:
    sandbox = LocalProcessSandbox()
    failures: list[str] = []
    for index, (title, code, expect) in enumerate(CASES, 1):
        result = sandbox.run(code, purpose=title,
                             timeout_s=expect.get("timeout_s", 30.0),
                             memory_mb=expect.get("memory_mb", 1024))
        problems: list[str] = []
        if result.ok != expect["ok"]:
            problems.append(f"期望 ok={expect['ok']}，实际 ok={result.ok}（{result.error}）")
        if expect.get("expect_error_type") and result.error_type != expect["expect_error_type"]:
            problems.append(f"期望 {expect['expect_error_type']}，实际 {result.error_type}（{result.error}）")
        if expect.get("expect_timeout") and not result.timed_out:
            problems.append("期望被判超时，实际没有")
        if expect.get("expect_error_any") and not result.error:
            problems.append("期望报错，实际没有")
        if expect.get("expect_stdout") and expect["expect_stdout"] not in result.stdout:
            problems.append(f"stdout 里没有 {expect['expect_stdout']!r}")
        if expect.get("expect_stdout_absent") and expect["expect_stdout_absent"] in result.stdout:
            problems.append(f"stdout 里不该出现 {expect['expect_stdout_absent']!r}，说明内存上限没兜住")
        if expect.get("expect_trace_mentions") and expect["expect_trace_mentions"] not in (result.traceback_text or ""):
            problems.append(f"traceback 里没有 {expect['expect_trace_mentions']!r}")
        for name in expect.get("expect_artifacts", []):
            if not any(a["path"] == name for a in result.artifacts):
                problems.append(f"产物里没有 {name}")

        status = "OK  " if not problems else "FAIL"
        print(f"[{status}] {index:2d}. {title}  （{result.elapsed_s}s）")
        if result.error:
            print(f"         错误：{result.error}")
        for problem in problems:
            print(f"         × {problem}")
            failures.append(f"{title}：{problem}")

    print()
    if failures:
        print(f"沙箱验收：{len(CASES) - len(failures)}/{len(CASES)} 通过，{len(failures)} 条未通过")
        return 1
    print(f"沙箱验收：{len(CASES)}/{len(CASES)} 全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
