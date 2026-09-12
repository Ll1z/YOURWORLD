# AGENTS.md

本文件是项目常驻约定。任何会话开始前先读它，再读 `HANDOFF.md` 了解完整背景与决策记录。

## 项目定位

- 名称：GeoAnalyst（工作目录 `YOURWORLD`）
- 目标：以个人学习为目的，跑通「地理空间分析 Agent」的完整前沿技术栈
- 不商业化、不做产品包装，重点是架构理解与可复现的工程实现
- 首个可验收能力：自然语言提问 → 自动发现数据 → 写代码算空间关系 → 空间自检 → 出图出报告
- 当前阶段：Stage 1 已完成。数据接入（北京五区边界 + OSM 切片 + 数据卡片）、三个 MCP Server（geo_catalog / geo_compute / geo_knowledge）、Agent 裸循环与空间自检器均已实现，端到端问答跑通并可复现。可视化层已接上：`web/` 的 FastAPI 薄壳 + Cesium 前端消费 `visual_hints.json`，飞相机、标命中、画半径圈与两点连线。范围见 `HANDOFF.md`

## 沟通约定

- 用中文回复，先给结论再给依据
- 不写无必要的注释，不引入此前未讨论过的依赖
- 每个阶段结束时给出一条可运行的验收命令
- 涉及取舍时列出备选方案与理由，不做默认沉默决策

## 技术栈

- Python 3.12，用 `uv` 管理解释器与虚拟环境。本机直连 pypi.org 会被 TLS 干扰（握手 EOF），`pyproject.toml` 里把清华镜像设为默认 index，否则依赖装不上；lock 里的包地址随之指向镜像
- 空间计算：DuckDB（spatial / FTS / vss 扩展）、GeoPandas、Shapely 2、pyogrio
- MCP：官方 Python SDK `mcp` 2.x（`from mcp.server.mcpserver import MCPServer`；FastMCP 在 2.x 已更名为 MCPServer，`mcp.server.fastmcp` 路径会直接报错），先 stdio，稳定后 Streamable HTTP
- MCP 客户端（实测坑）：用 `mcp.client.stdio.stdio_client` + `ClientSession`，**进入上下文后必须显式 `await session.initialize()`**，否则服务端对后续请求一律返回 `Invalid request parameters`；返回模型字段全是 snake_case（`server_info` / `protocol_version` / `resource_templates` / `structured_content`）。冒烟测试见 `scripts/mcp_smoke_test.py`
- 检索：单文件索引 `data/processed/knowledge.duckdb`（由 `scripts/build_knowledge_index.py` 构建），DuckDB FTS(BM25) + VSS(HNSW, cosine) 双路召回后按 RRF(k=60) 融合。中文没有可用分词器，自己按字符 bigram 切（Lucene CJKBigramFilter 的做法），ASCII 按词切；embedding 用 fastembed + `BAAI/bge-small-zh-v1.5`（512 维，ONNX CPU，不需要 torch），权重缓存在 `data/models/fastembed/`，HuggingFace 端点走 hf-mirror。dense 侧必须双阈值（`MIN_SIM` / `SIM_MARGIN`）：bge 对无关文本的相似度基线偏高（实测完全无关的问句也有 0.46），不设阈值会让无关语料块塞满 top-k 并稀释 RRF
- 沙箱：L1 受限子进程（`geo_compute/sandbox.py`）。Windows 侧用 Job Object 压内存与进程数，子进程内装导入黑名单 + socket 守卫 + 路径围栏，执行前先过 AST 静态检查；后端可替换，换 L2 Docker 只动这一个类
- 模型：DeepSeek API（OpenAI 兼容接口），密钥只放 `.env`
- 观测：`agent/telemetry.py` 把一次运行落成一棵 OTel span 树——invoke_agent 根 span 下挂 chat（每次模型调用）与 execute_tool（每次工具调用）子 span，属性名走 GenAI 语义约定（`gen_ai.*`），项目自己的属性统一 `geo.` 前缀。出口两个：本地 `spans.jsonl`（每次运行必写，离线可查）与 OTLP/HTTP（设了 `OTEL_EXPORTER_OTLP_ENDPOINT` 才启用，Langfuse v3 走这个口）。`.env` 里的观测配置由 `Settings.apply_otel_env()` 倒进环境变量：pydantic-settings 读 `.env` 只填字段、不写 `os.environ`，而 OTLP exporter 只认环境变量

## 数据来源

- 分析计算：OSM（openstreetmap.fr 北京省级切片，WGS84；可离线、可落盘、无条款限制）
- 可视化底图与影像：天地图（CGCS2000，与 WGS84 厘米级一致，国内加载快）。Key 是「浏览器端」类型，只能注入页面由浏览器直连，服务端代理会被拒（403 / code 301012）；Cesium 的 WMTS 层级要按 `tileMatrixLabels` 错位映射，天地图 `w` 矩阵集 id=1 就是 2×2 瓦片
- 对照与补齐：DataV.GeoAtlas（GCJ-02，必须先纠偏才能参与空间运算）
- 禁止：高德 / 百度数据直接用于空间计算或落盘（坐标加密 + 条款限制）

## 硬性规则

- 数值结论（面积、距离、数量、排名）必须由代码算出，LLM 不得直接生成数字
- 任何空间运算先显式声明 CRS；禁止拿经纬度直接算几何量。面积一律用大地线面积（`pyproj.Geod.geometry_area_perimeter`，椭球面精确、无投影变形）；距离与缓冲区用 UTM 50N（EPSG:32650，中央经线 117°E 覆盖北京）
- 半径查询必须先有明确、可命名的中心，和路径规划要先选起点同理：中心要么由调用方给坐标，要么由 `find_places` 解析地名得到。**禁止拿行政区几何代表点（point_on_surface）当圆心**——海淀区实测，以几何代表点为圆心问「800 米内有哪些便利店」会得到 0 个，那是几何产物不是事实。口径见 `servers/geo_knowledge/anchor_scope.json`
- 类别查询不设默认类别：`query_nearby` / `summarize_poi` 的类别必须由调用方给定，认不出来就报错并给近似建议，**绝不返回 0 条**（0 条要留给「真的没有」）。可查类别是库内真实存在的 `(category_key, category_value)` 组合（469 个，见 `compute://categories`），中文说法（高校 / 药店 / 公园…）在 `servers/geo_knowledge/categories/aliases.json`，由 `query.expand_category` 单点展开——默认值会把「想查 A 却拿到 B」变成静默替换。另有 15 个只存在于 anchor 层的组合（`railway=station` 等）单列在 `compute://categories` 的 `anchor_layer_only` 里
- 每个数据集必须配一张数据卡片（dataset card），字段规范见 `HANDOFF.md`
- MCP 分工：Resource 承载上下文（数据卡、schema、字典），Tool 承载动作与计算，不把一切都做成 Tool
- 工具返回的明细必须有上限：计数保持完整、明细按距离取前 N 条，并显式标注 `hits_truncated`。模型够不着的明细等于不存在——实测一次 10 公里半径的高校查询命中 158 条、37920 字符，回喂时被截到 20000，模型拿不到全量反而跑去沙箱里折腾了好几步
- 空间对象查询走空间索引（R-tree / H3 网格），不用向量检索做空间过滤
- 图层的口径边界必须显式：`anchor` 层（地铁站、火车站、公交站、地名锚点）默认不参与半径查询——它回答「在哪」，POI 两层回答「有什么」。查站点类设施才传 `query_nearby(include_anchor=true)`，命中单独计在 `count_anchor`，不并进默认口径；类别只在 anchor 层时默认查询直接报错并指出这个开关，不给一个会被当成结论的 0 条
- 知识检索只有一套口径：`search_knowledge` / `search_datasets` 走混合检索，索引缺失时直接报错，**不得静默退回关键词匹配**——换一种检索方式就是换了一套口径
- Agent 生成的代码一律在 `sandbox/run_<id>/` 内以受限子进程执行：禁网、超时（默认 30 秒）、内存上限（默认 1024 MB）、数据只读、只许写运行目录。`run_python` 是它的对外入口，也是唯一入口——不要在别处起子进程跑 Agent 写的代码
- 能用现成工具回答的，不许用 `run_python` 绕过去：工具的口径全项目唯一，脚本里的口径是临时的
- 敏感信息只放 `.env`，禁止写入代码或提交进版本库
- 观测不得成为跑通的前提：SDK 缺失或出口配错只记一条告警，不中断问答
- 多 Agent 只做加法：复核者只判断、不改写回答、不产出任何数字，判重做的标准是封闭的四条（答非所问 / 结论无工具支撑 / 该说明的没说明 / 有子问题没答）；重做最多一次，且只有在复核问题更少时才顶替原答案——复核者的意见是线索，不是命令
- 经验库只放能追到出处的条目：已确认的进 lessons（会被索引成 kind=experience，检索得到），`scripts/distill_experience.py` 从运行轨迹抽出的候选留在 candidates 里——不索引、不当结论用。

## 目录约定

Stage 1 骨架已于 2026-09-12 建立，当前结构：

```
servers/geo_catalog/           MCP：数据目录 + 混合检索；数据卡片存 cards/，索引实现在 index.py，embedding 封装在 embedding.py
servers/geo_compute/           MCP：DuckDB-spatial + 沙箱执行 + 出图
servers/geo_knowledge/         MCP：标准 / 术语 / 方法库；坐标系口径与纠偏在 coords/，类别中文别名在 categories/，程序性记忆（经验库）在 experience/lessons.json
agent/                         mcp_hub（聚合三个 MCP Server）+ loop（裸循环）+ selfcheck（空间自检）+ team（多 Agent：规划 / 复核）+ telemetry（OTel span）+ config
web/                           FastAPI 薄壳（server.py）与 Cesium 前端（index.html），只消费 agent/report.py 的产物
sandbox/                       代码执行运行目录（每次运行一个 run_<id> 子目录，留 code / _job.json / _result.json / 产物，可回放；不入库）
eval/                          评测集：cases.json 题库、ground_truth.json 冻结期望值、__init__.py/executor.py 比对与判定、runs/ 每次跑的产物（不入库）
data/                          样例数据与索引；raw 与 processed 默认不入库，地基数据走 .gitignore 白名单。data/models/ 放本地 embedding 权重（可重新下载，不入库）
scripts/                       一次性数据准备、验证与评测脚本；类别别名对库校验见 check_categories.py，检索索引见 build_knowledge_index.py，评测入口见 build_eval.py / run_eval.py
HANDOFF/                       项目交接包（HANDOFF.md、原始会话记录、打包 zip）
```

servers/ 下的包以可编辑模式安装（hatchling），全项目可直接调用，例如 `from geo_knowledge.coords.gcj02 import gcj02_to_wgs84`。

## 运行入口

数据链路（按顺序执行）：

1. `scripts/build_district_boundaries.py` —— 生成五区 WGS84 行政边界，并按需补齐西城区
2. `scripts/build_poi.py` —— 从 OSM 切片抽取 POI 点层与面层，标注所属区
3. `scripts/build_duckdb.py` —— 装载 DuckDB 并建立 R-tree 空间索引
4. `scripts/ask_nearby.py` —— 端到端查询，产出 result.geojson / result.csv / query.sql / report.md
5. `scripts/build_knowledge_index.py` —— 从数据卡片、口径文件、类别别名、坐标系定义与经验条目切块，建 `knowledge.duckdb` 的 FTS 与 HNSW 索引
6. `scripts/distill_experience.py` —— 扫运行轨迹，把重复出现的坑抽成候选经验并累加次数（`--dry-run` 只看不写）
7. `scripts/mcp_smoke_test.py` —— stdio 拉起三个 MCP Server，全量校验 tools / resources / resource templates 与工具调用
8. `scripts/sandbox_smoke_test.py` —— 沙箱验收：正常执行、静态检查、禁网、禁起进程、文件围栏、超时、内存上限、traceback 回传，逐条真起子进程验证
9. `scripts/ask_agent.py "问题"` —— Agent 裸循环：自然语言 → 选 MCP 工具 → 答案 + 空间自检 + visual_hints（前端消费）；加 `--team` 换成多 Agent 模式（规划者 → 执行者 → 复核者）
10. `scripts/build_eval.py` —— 跑真实工具生成/刷新 `eval/ground_truth.json`，并做 cross_check 一致性断言
11. `scripts/run_eval.py` —— 评测：`--mode data` 零 token 逐字段比对冻结值；`--mode agent` 真跑 Agent；`--mode team` 跑多 Agent，判定口径与前两者相同，用来做对照
12. `web/server.py` —— Web 入口：`uv run python web/server.py --port 8000`，浏览器打开 `http://127.0.0.1:8000`

评测集是这一层的回归网。题库 `eval/cases.json` 每条用例都钉死查询中心（`center_ref` 指向具体锚点——同名候选相距数百米，会改变半径边缘设施的进出）；期望值不手写，由 `scripts/build_eval.py` 跑真实工具生成并冻结，`fingerprint` 里带 query.py / server.py / poi_scope.json / aliases.json 的哈希与 DB 大小、mtime，任何口径、别名或数据漂移都会显形。改动查询口径、别名表或数据之后，必须重跑 `build_eval.py` 再跑 `run_eval.py --mode data`。`--mode agent` 另外判定四件事：调了哪个工具、数字能否溯源、关键数字是否出现在答案里、该拒绝的是否拒绝；用例标 `agent_skip` 表示「两种正确行为无法用固定关键词区分」，只在 data 层断言。

Agent 循环跑在 `web/server.py` 里的独立后台事件循环中：MCP stdio 会话绑定在创建它的 loop 上，且 OpenAI SDK 同步阻塞，直接跑在 web 的 loop 上会让一次提问把静态页面一起卡住。天地图 Key 只在渲染 `index.html` 时注入，不落盘、不入库。

中心点的选择权留在界面上：`GET /api/places` 不经过 LLM 直接检索地名候选，候选同时画到地图与列表，两处都可点选。点选结果作为一条 clarification 随问题一起送给 Agent，并写进 `report.md` 与 `trace.json`——事后的报告里能看出「当时在几个同名地点里选了哪一个」。数值溯源把用户输入与 Resource 上下文、工具返回并列为可信来源，否则用户自己给的坐标会被判成幻觉。

查询口径只有一处实现：`servers/geo_compute/query.py`。MCP Tool `query_nearby` 与 `scripts/ask_nearby.py` 都调用它，禁止在别处重写 SQL。类别的解析与展开（含中文别名、`key=value`、近似建议）同样只在 `query.resolve_categories` / `query.expand_category` 一处，服务端只做 `ToolError` 包装。

空间自检器（`agent/selfcheck.py`）做五项检查：CRS、单位、几何有效性、量级自洽，以及**数值溯源**——最终回答里的每个数字都必须能在工具返回中找到出处。调用 `distance_between` 时额外做一次距离互证：UTM 平面距离与椭球面大地线距离必须互相印证（容差按舍入误差推导）。溯源不通过时循环会把回答打回重写（最多 2 轮）。实测模型确实会在叙述里自行估算两个设施之间的距离，只在提示词里禁止是不够的。

辅助脚本：`calibrate_gcj02.py`（纠偏算法校准）、`check_boundaries.py`（边界完整性体检）、`analyze_poi_overlap.py`（点面重复量化）、`check_categories.py`（类别别名表逐条对库校验）
