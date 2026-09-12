# AGENTS.md

本文件是项目常驻约定。任何会话开始前先读它，再读 `HANDOFF.md` 了解完整背景与决策记录。

## 项目定位

- 名称：GeoAnalyst（工作目录 `FAKNEWS`，目录名沿用了早期想法，与本项目主题无关）
- 目标：以个人学习为目的，跑通「地理空间分析 Agent」的完整前沿技术栈
- 不商业化、不做产品包装，重点是架构理解与可复现的工程实现
- 首个可验收能力：自然语言提问 → 自动发现数据 → 写代码算空间关系 → 空间自检 → 出图出报告
- 当前阶段：Stage 1 进行中。数据接入已完成（北京五区边界 + OSM 切片 + 数据卡片）；三个 MCP Server（geo_catalog / geo_compute / geo_knowledge）已实现并通过 stdio 冒烟测试；Agent 循环与空间自检器待做。范围见 `HANDOFF.md`

## 沟通约定

- 用中文回复，先给结论再给依据
- 不写无必要的注释，不引入此前未讨论过的依赖
- 每个阶段结束时给出一条可运行的验收命令
- 涉及取舍时列出备选方案与理由，不做默认沉默决策

## 技术栈

- Python 3.12，用 `uv` 管理解释器与虚拟环境
- 空间计算：DuckDB（spatial / FTS / vss 扩展）、GeoPandas、Shapely 2、pyogrio
- MCP：官方 Python SDK `mcp` 2.x（`from mcp.server.mcpserver import MCPServer`；FastMCP 在 2.x 已更名为 MCPServer，`mcp.server.fastmcp` 路径会直接报错），先 stdio，稳定后 Streamable HTTP
- MCP 客户端（实测坑）：用 `mcp.client.stdio.stdio_client` + `ClientSession`，**进入上下文后必须显式 `await session.initialize()`**，否则服务端对后续请求一律返回 `Invalid request parameters`；返回模型字段全是 snake_case（`server_info` / `protocol_version` / `resource_templates` / `structured_content`）。冒烟测试见 `scripts/mcp_smoke_test.py`
- 检索：DuckDB FTS + VSS 单文件索引起步，本地 bge 系列模型做 embedding 与 rerank
- 模型：DeepSeek API（OpenAI 兼容接口），密钥只放 `.env`
- 观测：OpenTelemetry GenAI 语义约定 + Langfuse（Stage 2 起接入）

## 数据来源

- 分析计算：OSM（openstreetmap.fr 北京省级切片，WGS84；可离线、可落盘、无条款限制）
- 可视化底图与影像：天地图（CGCS2000，与 WGS84 厘米级一致，国内加载快）
- 对照与补齐：DataV.GeoAtlas（GCJ-02，必须先纠偏才能参与空间运算）
- 禁止：高德 / 百度数据直接用于空间计算或落盘（坐标加密 + 条款限制）

## 硬性规则

- 数值结论（面积、距离、数量、排名）必须由代码算出，LLM 不得直接生成数字
- 任何空间运算先显式声明 CRS；禁止拿经纬度直接算几何量。面积一律用大地线面积（`pyproj.Geod.geometry_area_perimeter`，椭球面精确、无投影变形）；距离与缓冲区用 UTM 50N（EPSG:32650，中央经线 117°E 覆盖北京）
- 每个数据集必须配一张数据卡片（dataset card），字段规范见 `HANDOFF.md`
- MCP 分工：Resource 承载上下文（数据卡、schema、字典），Tool 承载动作与计算，不把一切都做成 Tool
- 空间对象查询走空间索引（R-tree / H3 网格），不用向量检索做空间过滤
- Agent 生成的代码一律在 `sandbox/run_<id>/` 内以受限子进程执行：禁网、超时、内存上限、数据只读
- 敏感信息只放 `.env`，禁止写入代码或提交进版本库

## 目录约定

Stage 1 骨架已于 2026-09-12 建立，当前结构：

```
servers/geo_catalog/           MCP：数据目录 + 混合检索；数据卡片存 cards/
servers/geo_compute/           MCP：DuckDB-spatial + 沙箱执行 + 出图
servers/geo_knowledge/         MCP：标准 / 术语 / 方法库；坐标系口径与纠偏在 coords/
agent/                         planner + 检索路由 + 循环 + 空间自检器
sandbox/                       代码执行运行目录（每次运行独立子目录）
eval/                          任务集 + 指标 + 消融实验
data/                          样例数据与索引；raw 与 processed 默认不入库，地基数据走 .gitignore 白名单
scripts/                       一次性数据准备与验证脚本
HANDOFF/                       项目交接包（HANDOFF.md、原始会话记录、打包 zip）
```

servers/ 下的包以可编辑模式安装（hatchling），全项目可直接调用，例如 `from geo_knowledge.coords.gcj02 import gcj02_to_wgs84`。

## 运行入口

数据链路（按顺序执行）：

1. `scripts/build_district_boundaries.py` —— 生成五区 WGS84 行政边界，并按需补齐西城区
2. `scripts/build_poi.py` —— 从 OSM 切片抽取 POI 点层与面层，标注所属区
3. `scripts/build_duckdb.py` —— 装载 DuckDB 并建立 R-tree 空间索引
4. `scripts/ask_nearby.py` —— 端到端查询，产出 result.geojson / result.csv / query.sql / report.md
5. `scripts/mcp_smoke_test.py` —— stdio 拉起三个 MCP Server，全量校验 tools / resources / resource templates 与工具调用

查询口径只有一处实现：`servers/geo_compute/query.py`。MCP Tool `query_nearby` 与 `scripts/ask_nearby.py` 都调用它，禁止在别处重写 SQL。

辅助脚本：`calibrate_gcj02.py`（纠偏算法校准）、`check_boundaries.py`（边界完整性体检）、`analyze_poi_overlap.py`（点面重复量化）
