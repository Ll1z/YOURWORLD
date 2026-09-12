"""Agent 运行时配置。

敏感信息只从 .env 读取，不写进代码、不进版本库。
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    deepseek_api_key: str
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"
    request_timeout_s: float = 180.0
    temperature: float = 0.0
    # 可视化：天地图 Key 类型是「浏览器端」，只能注入页面由浏览器直连（服务端代理会被拒）
    tianditu_key: str = ""
    cesium_base_url: str = "https://cdn.jsdelivr.net/npm/cesium@1.135.0/Build/Cesium/"
    # 观测出口（可选）：留空时只写本地 spans.jsonl，配了就同时推到 OTLP 后端
    otel_exporter_otlp_endpoint: str = ""
    otel_exporter_otlp_headers: str = ""

    def apply_otel_env(self) -> None:
        """把 .env 里的观测配置倒进环境变量。

        pydantic-settings 读 .env 只是填进字段，不会写进 os.environ，
        而 OTLP exporter 只认标准环境变量——少这一步，写在 .env 里的出口等于没配。
        已存在的环境变量优先，不覆盖进程级配置。
        """
        for key, value in (
            ("OTEL_EXPORTER_OTLP_ENDPOINT", self.otel_exporter_otlp_endpoint),
            ("OTEL_EXPORTER_OTLP_HEADERS", self.otel_exporter_otlp_headers),
        ):
            if value and not os.getenv(key):
                os.environ[key] = value

    @property
    def openai_base_url(self) -> str:
        """把 base_url 归一成 OpenAI 兼容的 /v1 形式，兼容写与不写 /v1 两种配置。"""
        base = self.deepseek_base_url.rstrip("/")
        return base if base.endswith("/v1") else f"{base}/v1"
