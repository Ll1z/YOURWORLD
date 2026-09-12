# HANDOFF.md · 项目上下文交接包

> 导出时间：2026-09-12（Asia/Shanghai）
> 来源设备：旧笔记本（Python 3.9.6 / RTX 3060 Laptop 6GB）
> 来源线程：`01a09484-a905-7b93-9144-8fd3060a79ea`
> 去向设备：新笔记本（Core Ultra 9 275HX + RTX 5080 Laptop 16GB，环境干净）
> 用途：在新设备上无缝接续本项目，不丢失任何已定决策

## 0. 怎么用这份交接包

1. 先读 `AGENTS.md`（常驻约定，Codex 会自动加载），再读本文件（完整背景与决策）
2. 想恢复原始对话：把 `handoff/codex-session-01a09484-a905-7b93-9144-8fd3060a79ea.jsonl` 复制到新设备的 `%USERPROFILE%\.codex\sessions\2026\09\12\`，重启 Codex 后应能在列表里看到这条线程
3. 想直接开新会话：把第 9 节的「恢复提示词」整段粘给新会话
4. 恢复后先让 Codex 复述理解，确认无误再开工

## 1. 项目是什么

- 一个以**个人学习**为目的的 Agent 项目，不商业化、不做产品化包装
- 主题：**地理空间分析 Agent**（代号 GeoAnalyst，工作目录 `FAKNEWS` 只是沿用了早期想法的目录名）
- 要跑通的技术栈：MCP + Agentic RAG + CodeAct（代码执行）+ 空间自检 + 轨迹观测
- 第一个可验收能力：用自然语言提问，Agent 自动发现数据、写代码算空间关系、自检、出图出报告
- 导出时状态：**尚未写任何代码**，目录里只有这份交接包（导出之后的实际进展见第 12 节）

## 2. 决策记录（按讨论顺序，含理由）

| # | 决策 | 理由 |
|---|---|---|
| 1 | 领域选地理信息，而非新闻核查 / Agent 评测平台 / 自研 runtime | LLM 空间推理天然弱，必须靠工具与代码兜底，Agent 价值真实；空间结果可自检、可评测，闭环天然成立 |
| 2 | 定位收敛为个人学习，不商业化 | 砍掉合规、审图号、责任边界、销售周期等噪音，聚焦技术 |
| 3 | 采用 CodeAct（Agent 写 Python 代码）而非固定工具集 | 空间操作空间近乎无限（投影、栅格、拓扑、插值），固定 tool schema 撑不住；代价是必须配沙箱 |
| 4 | MCP 拆三个 Server：geo-catalog / geo-compute / geo-knowledge | 职责单一、可独立演进；geo-catalog 管数据发现，geo-compute 管计算与出图，geo-knowledge 管标准与口径 |
| 5 | MCP primitive 严格分工 | Resource 放上下文（数据卡、schema、字典），Tool 放动作与计算，Prompt 放可复用分析模板，Sampling 用于服务端反查（如地名消歧），Elicitation 用于向用户补参数 |
| 6 | RAG 用四类索引，不搞单向量库打天下 | 元数据与文档用混合检索；schema 与 SQL 示例用结构化 few-shot；空间对象用 R-tree / H3 索引；影像用图像 embedding。用向量做空间过滤是反模式 |
| 7 | 检索策略：filter-then-rank | 先用 bbox、时间、分辨率、许可做结构化过滤，再做语义排序与 rerank |
| 8 | 铁律：数值结论必须由代码算出 | 面积、距离、数量、排名绝不能由 LLM 生成，RAG 只负责给模型喂上下文 |
| 9 | 数据卡片（dataset card）是系统的原子单元 | 卡里必须写 CRS、单位、时间范围、许可、样例行、已知坑（尤其 GCJ-02 / BD-09 偏移问题） |
| 10 | 沙箱分两层：L1 受限子进程起步，L2 Docker 可选升级 | 个人学习场景 99 的问题是防事故而非防攻击；手写 L1 有教学价值；执行器后端设计成可替换 |
| 11 | 硬件结论：新设备更强，但对 Stage 1 无影响 | 主循环推理在云端 API；embedding 与 rerank 在 6GB 显存上已富余；升级收益集中在本地大模型、遥感深度学习、多服务并发 |
| 12 | 资源方案：DeepSeek API + 北京 + 允许联网 + 零 Key 数据源 | 避免把时间耗在配额、授权、审图号上；OSM 是 WGS84，天然绕开国内坐标系偏移坑 |

## 3. 架构设计

### 3.1 主流程

```
用户提问
  → Planner：抽取意图 + 空间约束（地名 / bbox / 时间 / 单位 / 指标口径）
  → 检索路由
       元数据与文档 → 混合检索（BM25 + dense + rerank）
       空间对象     → 空间索引（bbox + H3 粗筛，再精确几何）
  → 规划并执行：经 MCP 调工具 或 沙箱跑代码
  → 空间自检：CRS、单位、几何有效性、量级合理性
  → 失败则读 traceback → 检索错误修复经验库 → 重写重跑
  → 出图 + 报告（含数据来源、CRS、时间、口径、局限）
  → 轨迹落库 → 蒸馏为经验与数据卡 → 回灌 RAG
```

### 3.2 MCP 三个 Server

- `geo-catalog`：STAC、OSM Overpass、本地文件清单；对外提供数据卡 Resource 与检索 Tool
- `geo-compute`：DuckDB-spatial 查询、沙箱代码执行、地图渲染
- `geo-knowledge`：标准规范、术语词典（如「建成区」「可达性」口径）、分析方法库
- 传输：先 stdio，用 `mcp dev` 的 Inspector 调试；稳定后换 Streamable HTTP + OAuth，让 Codex、Claude Desktop 等客户端都能直连

### 3.3 RAG 四类索引

| 检索对象 | 索引形态 | 检索方式 |
|---|---|---|
| 数据卡片 / 文档 / 标准 | BM25 + dense 混合 | filter-then-rank，先结构化过滤再语义排序 + rerank |
| 表 schema / SQL 示例 | 结构化 + few-shot 四元组 | （问题, schema, SQL, 结果形态）按相似度取用 |
| 空间对象本身 | R-tree / H3 网格 | 绝不用 embedding 做空间过滤 |
| 地图 / 影像切片 | 图像 embedding（CLIP 类） | 以图搜地、找相似地块（Stage 3 可选） |

### 3.4 数据卡片规范（新数据集必须建卡）

字段：`id`、名称、描述、空间范围 bbox、时间范围、分辨率 / 精度、CRS、字段 schema 与语义、单位、更新频率、来源与许可、样例行、已知坑。

已知坑的必填项示例：坐标系归属（WGS84 / GCJ-02 / BD-09）、面积单位陷阱、行政边界版本年份。

### 3.5 沙箱设计

- L1（Stage 1 采用）：`sandbox/run_<id>/` 独立运行目录；受限子进程；超时 + 内存上限 + 进程数限制；数据只读挂载；沙箱环境只装最小依赖（geopandas、duckdb 等），不装网络库；对生成代码做 AST 静态检查，拦截 `os.system`、`subprocess`、`shutil.rmtree` 之类调用；执行记录（代码 / 输出 / 产物哈希）落库可回放
- L2（Stage 3 可选）：Docker 容器，内核级 namespace + cgroups，网络与文件系统隔离
- 要求：执行器写成可替换后端，换层时只改一个模块

### 3.6 经验库（程序性记忆）

成功轨迹蒸馏为「分析配方」（步骤链 + SQL 模板 + 参数），失败修复蒸馏为「错误模式 → 修法」对；下次相似问题先检索配方再动手。这是让系统越用越准的关键。

## 4. 环境与资源

### 4.1 旧设备实测（已完成探测）

- 有：git 2.45.1、Python 3.9.6、RTX 3060 Laptop（6GB）
- 缺：Python 3.11+、uv、Docker、Ollama、WSL 发行版
- 结论：Python 3.9 过老，现代 GIS 与 Agent 库装不上，必须升级

### 4.2 新设备

- CPU：Core Ultra 9 275HX（24 核），GPU：RTX 5080 Laptop 16GB GDDR7
- 环境干净，尚无 Codex，需要先安装并登录
- 16GB 显存的意义：可常驻 bge-m3 级向量模型、可跑 14B 量化本地模型、可做小模型遥感微调

### 4.3 新设备准备清单（Step 0）

1. 安装 Codex（桌面版或 CLI），用同一账号登录
2. 安装 git
3. 安装 `uv`（用于装独立 Python 3.12，不需要管理员权限）
4. 可选：Docker Desktop（若愿意装，可跳过 L1 直接用容器沙箱）
5. 预留磁盘：首次数据与模型下载约 8 到 10GB

## 5. 数据源清单（零 Key 优先）

- OSM 底表：Geofabrik 全国包（约 1.5GB）或 BBBike 城市级切片（数百 MB），转 GeoPackage 后完全离线可用；医院、学校、便利店、道路、建筑、边界都在里面
- 行政边界：DataV.GeoAtlas 的省市区县 GeoJSON（免注册）
- 人口格网：WorldPop 100m（可选，做可达性分析）
- 卫星影像：Copernicus Data Space 或 Planetary Computer（免费注册，Stage 2 起可选）
- 中文 POI 增强：天地图 / 高德个人开发者 Key（可选，仅当 OSM 中文质量不满意时启用）
- 合规：个人学习、不公开分发，无需处理审图号

## 6. Stage 1 范围与验收标准

### 范围

- 数据：北京市 2 到 3 个区县的 OSM 切片 + 行政边界 + 数据卡片
- MCP：`geo-catalog` 与 `geo-compute` 两个 stdio Server
- 检索：DuckDB FTS + VSS 混合检索，返回候选数据集与验证方法（SQL 经验库先手工放几条种子）
- Agent：裸循环（不引入编排框架），工具调用 + 结构化输出 + 引用格式固定
- 沙箱：L1 受限子进程执行器
- 自检：CRS、单位、几何有效性、量级合理性四项基础检查

### 明确不做

- 多 Agent 协作、Docker 沙箱、可视化前端、经验库自动蒸馏、OTel 全链路观测（留给 Stage 2 与 Stage 3）

### 验收标准

- 能用自然语言回答：北京市 X 区有哪些医院在 1 公里内（直线距离）
- 回答必须附：数据来源、CRS 声明、计算方法、可复现脚本、结果落盘文件
- 换一个同类问题（如某区内便利店分布）无需改代码即可回答

## 7. 里程碑

- 第 1 周：裸循环 + 最小空间工具，能回答最简单的「某点周围有什么」—— 已完成
- 第 2 周：CodeAct 沙箱 + traceback 自修复 + 空间自检器 —— 已完成（沙箱 L1 + 五项空间自检 + traceback 回传重跑）
- 第 3 周：多源数据接入（OSM + 本地 + 边界）+ 出图与报告 —— 已完成（出图即 Cesium 前端）
- 第 4 周：评测集（30 到 50 题 + ground truth）+ 看板 + MCP 封装 + 消融实验 —— 评测集已完成（43 题）＋ MCP 封装已完成；看板与消融实验未开始

## 8. 未决问题（新会话开始时确认）

- DeepSeek 具体模型与档位（拿到 Key 后确认，模型名写进 `.env`）
- 北京样例数据范围：先做哪 2 到 3 个区县
- embedding 选型（已定）：`fastembed` + `BAAI/bge-small-zh-v1.5`，512 维，ONNX CPU，不需要 torch。bge-m3 等更大模型留给后续消融对比
- 是否安装 Docker Desktop（装了就能直接上 L2 沙箱）
- 观测方案：OTel + Langfuse 还是自建 SQLite 轨迹表（Stage 2 再定）

## 9. 恢复提示词（整段复制给新设备上的 Codex）

```
我在新设备上继续 FAKNEWS 项目（地理空间分析 Agent，个人学习用途）。

请先完整阅读 AGENTS.md 和 HANDOFF.md，然后用中文向我复述：
1) 项目目标与当前阶段
2) 已确定的关键决策（尤其沙箱、MCP 分工、RAG 四类索引、数值必须由代码产出这条铁律）
3) Stage 1 的范围、明确不做的部分、验收标准
4) 你建议的第一步动作与理由

复述完先等我确认，不要立刻开始写代码。
```

## 10. 迁移清单

要拷贝：

- 整个 `FAKNEWS` 目录（含 `AGENTS.md`、`HANDOFF.md`、`handoff/`）

可选拷贝（用于恢复原始对话）：

- `handoff/codex-session-01a09484-a905-7b93-9144-8fd3060a79ea.jsonl`
- 目标位置：新设备 `%USERPROFILE%\.codex\sessions\2026\09\12\`
- 前提：同一账号登录；若 Codex 未识别，改用第 9 节提示词开新会话即可，上下文不会丢

不要拷贝：

- `%USERPROFILE%\.codex\auth.json`（登录凭据，新设备重新登录）
- `.codex` 下的各类 sqlite（设备状态，拷过去可能引发不一致）

## 11. 业务参照（仅作设计参照，不是本项目目标）

早期讨论过的商业形态，留档备查：面向城市规划、零售选址、保险风控、物流网络、环保应急的「地理分析交付引擎」，交付物是报告 + 地图 + 数据表 + 可复现脚本；护城河是数据卡片质量、经验库、行业口径库、评测集。

本项目**不追求**上述商业化能力，仅在架构上留出可扩展的位置。

## 12. 进展追加（导出之后的实际进展）

> 本节由新设备上的会话追加，只记「与上面这份快照的差异」。实时约定一律以 `AGENTS.md` 为准。

### 环境落定

- 解释器改用 `uv` 管理（Python 3.12），不再依赖系统 Python 3.9.6
- 工作目录定为 `YOURWORLD`（旧的 `FAKNEWS` 目录名只存在于这份快照里）
- 模型：DeepSeek API（OpenAI 兼容接口），密钥只放 `.env`
- 数据源分工：OSM openstreetmap.fr 北京省级切片（BBBike 五区切片）承担空间计算；天地图只做可视化底图（浏览器端 Key，服务端代理会被 403 / code 301012）

### 已完成

- 数据：五区（东城 / 西城 / 海淀 / 朝阳 / 丰台）行政边界 + OSM 点面两层 POI，装载进单文件 `data/processed/geo.duckdb` 并建 R-tree 空间索引；五区大地线面积合计约 1294 km²
- MCP：`geo_catalog` / `geo_compute` / `geo_knowledge` 三个 Server 全部就位，合计 10 个 Tool、7 个 Resource（另有 2 个 Resource 模板）；`scripts/mcp_smoke_test.py` 全量校验零失败
- Agent：裸循环（`agent/loop.py`）+ 五项空间自检器（CRS / 单位 / 几何有效性 / 量级自洽 / 数值溯源），溯源不通过会把回答打回重写
- Web：`web/` 的 FastAPI 薄壳 + Cesium 前端已消费 `visual_hints.json`，可飞相机、标命中、画半径圈与两点连线；中心点候选在界面与地图两处都能点选
- 沙箱与 CodeAct：`geo_compute/sandbox.py`（L1 受限子进程，后端可替换）+ `run_python` 工具 + `scripts/sandbox_smoke_test.py`（11 条护栏验收）；Agent 提示词加了「工具优先、现成工具拼不出来才写代码」「照 traceback 改、同一个错两次就停」两条
- 评测：`eval/` 评测集 43 题（nearby 19 / aggregate 11 / distance 6 / places 3 / refusal 2 / tool_error 2），期望值由真实工具生成后冻结；`--mode data` 零 token 回归，`--mode agent` 判工具选择、数值溯源、关键数字与拒答行为

### 与「明确不做」的差异

- **可视化前端**已经做了：原计划留给 Stage 2，实际在 Stage 1 内就用 Cesium 落地（只做消费端，不做分析端）
- 其余仍按原计划：多 Agent 协作、Docker L2 沙箱、经验库自动蒸馏、OTel 全链路观测都留给 Stage 2

### 一条值得写进快照的教训

空间自检器不是可选项。实测模型确实会在叙述里自行估算两个设施之间的距离，只在提示词里写「不要编数字」拦不住；必须让「答案里每个数字都能在工具返回里找到出处」成为机器判定。

### 另一条教训：溯源池不能装目录

溯源一开始把预加载的类别目录也当成可信来源，于是「4 家咖啡馆」这种凭空计数被目录里某个类别的 4 兜住——140 个与本次提问无关的数字，让小整数几乎必然「有出处」。现在目录型清单（类别目录、数据集清单、知识库索引规模）一律不进溯源池，单位换算的容差也按数值量级给（4326 ÷ 1000 不该把 4 判成有出处）。收紧之后重跑同一道题，模型放弃了自己数，改用 `run_python` 分档——规则和围栏是一起起作用的。

### ④-1 观测层落地

- 一次运行 = 一棵 OTel span 树：`invoke_agent` 根 span 下挂 `chat` 与 `execute_tool` 子 span，属性名走 GenAI 语义约定（`gen_ai.*`），项目自己的属性统一 `geo.` 前缀
- 出口两个：本地 `spans.jsonl`（每次运行必写，随报告一起落盘）与 OTLP/HTTP（设 `OTEL_EXPORTER_OTLP_ENDPOINT` 才启用；Langfuse v3 的入口是 `https://cloud.langfuse.com/api/public/otel` + Basic Auth 头）
- `.env` 里的观测配置由 `Settings.apply_otel_env()` 倒进 `os.environ`：pydantic-settings 读 `.env` 只填字段、不写环境变量，而 OTLP exporter 只认环境变量——少这一步，写在 `.env` 里的出口等于没配
- 观测不是跑通的前提：没装 SDK 或出口配错只记一条告警，报告里单列「观测」段
- 顺带发现：`gen_ai.response.model` 显示 DeepSeek 把 `deepseek-chat` 路由到了 `deepseek-flash`——这种「请求模型 ≠ 实际模型」以前是看不见的

### ③ 混合检索落地（知识库问答的数据侧）

- 语料切块（141 块）：数据卡按「整卡 + schema + 每条已知坑」切，口径文件按顶层小节切，`aliases.json` 的 80 条中文别名与 5 条「查不到」说明各算一块
- 索引：单文件 `data/processed/knowledge.duckdb`，FTS(BM25) 与 HNSW(cosine) 双路召回后按 RRF(k=60) 融合；构建入口 `scripts/build_knowledge_index.py`，冷启动到建完约 4 秒
- 中文分词：DuckDB FTS 不带中文分词器，直接索引中文几乎检索不到（测「医院」返回 0）。改按字符 bigram 切（Lucene CJKBigramFilter 的做法），ASCII 走 `[a-z0-9_]+`，检索立刻正常
- embedding：`fastembed` + `BAAI/bge-small-zh-v1.5`（512 维，ONNX CPU，不需要 torch），权重缓存在 `data/models/fastembed/`；本机 `huggingface.co` 不通（20 秒超时），改走 `hf-mirror.com`
- dense 侧双阈值：bge 对无关文本的相似度基线偏高（实测完全无关的问句也有 0.46），只设绝对阈值会让无关语料块塞满 top-20 并稀释 RRF。现在用 `dense_floor = max(MIN_SIM, top1 - SIM_MARGIN)`，两个阈值一起兜
- 工具面：`search_datasets`（只在数据卡类语料里检索）与 `search_knowledge`（全语料，可按 kinds / dataset_id 过滤）都返回带 `source_uri` 的命中，要全文就按 URI 读 Resource；索引没建好时直接报 ToolError，**不退回关键词匹配**——静默降级等于换了一套口径
- Agent 侧：`catalog://knowledge`（索引规模与构成）进预加载上下文，同时进溯源池排除名单；提示词加第 14 条，把「口径怎么定的 / 这个中文说法对应哪个 OSM 标签 / 某份数据有什么坑 / 某类别为什么查不到」指向 `search_knowledge`
