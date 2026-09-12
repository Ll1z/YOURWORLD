# AGENTS.md

本文件是项目常驻约定。任何会话开始前先读它，再读 `HANDOFF.md` 了解完整背景与决策记录。

## 项目定位

- 名称：GeoAnalyst（工作目录 `FAKNEWS`，目录名沿用了早期想法，与本项目主题无关）
- 目标：以个人学习为目的，跑通「地理空间分析 Agent」的完整前沿技术栈
- 不商业化、不做产品包装，重点是架构理解与可复现的工程实现
- 首个可验收能力：自然语言提问 → 自动发现数据 → 写代码算空间关系 → 空间自检 → 出图出报告
- 当前阶段：Stage 1（尚未开始编码），范围见 `HANDOFF.md`

## 沟通约定

- 用中文回复，先给结论再给依据
- 不写无必要的注释，不引入此前未讨论过的依赖
- 每个阶段结束时给出一条可运行的验收命令
- 涉及取舍时列出备选方案与理由，不做默认沉默决策

## 技术栈

- Python 3.12，用 `uv` 管理解释器与虚拟环境
- 空间计算：DuckDB（spatial / FTS / vss 扩展）、GeoPandas、Shapely 2、pyogrio
- MCP：官方 Python SDK（FastMCP），先 stdio，稳定后 Streamable HTTP
- 检索：DuckDB FTS + VSS 单文件索引起步，本地 bge 系列模型做 embedding 与 rerank
- 模型：DeepSeek API（OpenAI 兼容接口），密钥只放 `.env`
- 观测：OpenTelemetry GenAI 语义约定 + Langfuse（Stage 2 起接入）

## 硬性规则

- 数值结论（面积、距离、数量、排名）必须由代码算出，LLM 不得直接生成数字
- 任何空间运算先显式声明 CRS；面积与距离一律在米制投影下计算，禁止拿经纬度直接算
- 每个数据集必须配一张数据卡片（dataset card），字段规范见 `HANDOFF.md`
- MCP 分工：Resource 承载上下文（数据卡、schema、字典），Tool 承载动作与计算，不把一切都做成 Tool
- 空间对象查询走空间索引（R-tree / H3 网格），不用向量检索做空间过滤
- Agent 生成的代码一律在 `sandbox/run_<id>/` 内以受限子进程执行：禁网、超时、内存上限、数据只读
- 敏感信息只放 `.env`，禁止写入代码或提交进版本库

## 目录约定

Stage 1 骨架已于 2026-09-12 建立，当前结构：

```
servers/geo_catalog/     MCP：数据目录 + 混合检索
servers/geo_compute/     MCP：DuckDB-spatial + 沙箱执行 + 出图
servers/geo_knowledge/   MCP：标准 / 术语 / 方法库
agent/                   planner + 检索路由 + 循环 + 空间自检器
sandbox/                 代码执行运行目录（每次运行独立子目录）
eval/                    任务集 + 指标 + 消融实验
data/                    样例数据与索引（raw/ 与 processed/ 不入库）
HANDOFF/                 项目交接包（HANDOFF.md、原始会话记录、打包 zip）
```
