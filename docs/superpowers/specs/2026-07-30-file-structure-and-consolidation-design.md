# 项目文件结构重新组织与抗碎片化设计规范

**日期**：2026-07-30  
**分支**：refactor/code-compliance  
**状态**：Spec 待审阅  

---

## 一、背景与目标

根据最新的《项目编码规范》（`CLAUDE.md`），当前仓库存在以下两大核心结构问题：

1. **运行时文件与源码混杂**：
   - 动态 SQLite 数据库 `ket_partner.db` 在根目录与 `src/` 目录下重复出现。
   - 导出生成的 Markdown 学习报告（如 `src/learning_report_*.md`）与日志混在源码目录中。
2. **模块过度碎片化（违反 CLAUDE.md §4.6 CCP 共同封闭原则与 §10.1 浅层透传）**：
   - `src/flow/ket_partner/` 将同一领域子任务过度拆分为 18 个微型文件（如 `input_classifier.py` 1.5KB、`profile_summarizer.py` 1.8KB 等）。
   - 追踪主流程需穿越大量无独立复用价值的微型文件，增大了认知成本。
   - `tests/flow/ket_partner/` 同样存在微型测试文件散落的问题。

**目标**：
1. 建立标准的 `storage/` 运行时数据隔离目录，将数据库、导出报告与日志收拢在 Git 忽略的专用区域。
2. 对 `src/flow/ket_partner/` 进行抗碎片化重构，合并为 3 大高内聚领域模块及标准的 LangGraph 节点与图分层结构。
3. 调整 `tests/` 目录结构，使其与 `src/` 模块结构保持 1:1 映射。
4. 清理全仓库的死代码（Dead Code）与硬编码相对路径。

---

## 二、目标项目结构（精确到文件级）

```
英语/
├── CLAUDE.md
├── pyproject.toml
├── pytest.ini
├── ruff.toml
├── mypy.ini
├── .gitignore                      # 增加 storage/ 规则
│
├── data/                           # 静态受控数据目录 (Git 追踪)
│   └── KET_vocabulary.csv          # 词汇库种子 CSV 表
│
├── storage/                        # ★ 统一运行时存储目录 (Git 忽略)
│   ├── db/
│   │   ├── .gitkeep
│   │   └── ket_partner.db          # 默认 SQLite 数据库文件
│   ├── reports/
│   │   ├── .gitkeep
│   │   └── learning_report_*.md    # 导出归档的学习报告
│   └── logs/
│       ├── .gitkeep
│       └── app.log                 # 运行日志
│
├── src/                            # 后端源码目录
│   ├── __init__.py
│   ├── api/                        # Web API 接口层 (Composition Root)
│   │   ├── __init__.py
│   │   ├── app.py
│   │   ├── deps.py
│   │   ├── llm_key.py
│   │   ├── main.py
│   │   ├── schemas.py
│   │   ├── settings.py             # DB_PATH 默认指向 storage/db/ket_partner.db
│   │   └── routes/
│   │       ├── __init__.py
│   │       ├── chat.py
│   │       ├── llm_key.py
│   │       └── report.py
│   ├── cli/                        # 命令行工具 (Composition Root)
│   │   ├── __init__.py
│   │   └── ket_partner/
│   │       ├── __init__.py
│   │       ├── chat_logger.py      # 日志调整至 storage/logs/
│   │       ├── commands.py
│   │       └── main.py             # DEFAULT_DB 默认指向 storage/db/ket_partner.db
│   ├── flow/                       # Agent 核心逻辑层
│   │   ├── __init__.py
│   │   ├── common.py               # 清理 dead code (IS_RUNNING_IN_PYTEST, llm_plus 等)
│   │   └── ket_partner/
│   │       ├── __init__.py
│   │       ├── agent.py            # 仅保留 KETPartnerAgent 统一入口包装
│   │       ├── config.py
│   │       ├── graph.py            # build_agent, 图拓扑与 route_* 函数
│   │       ├── nodes.py            # ★ NEW: 集中存放所有节点函数 (纯 State 读写薄壳)
│   │       ├── persistence.py      # 存放 KETPartnerRepos 等 Protocol 定义
│   │       ├── state.py
│   │       ├── sentence_domain.py  # ★ NEW(合并): 造句与校验全流程领域模块
│   │       ├── vocab_domain.py     # ★ NEW(合并): 选词、查释义与掌握度全流程领域模块
│   │       └── dialogue_domain.py  # ★ NEW(合并): 意图分类、对话分析与格式化领域模块
│   ├── persistence/                # 独立持久化层
│   │   ├── __init__.py
│   │   ├── bootstrap.py            # 默认初始化路径指向 storage/db/ket_partner.db
│   │   ├── migration.py
│   │   ├── models.py
│   │   ├── repos.py
│   │   └── schema.py
│   └── reporting/                  # 独立报告导出层
│       ├── __init__.py
│       └── ket_partner/
│           ├── __init__.py
│           ├── categories.py
│           ├── exporter.py         # 默认导出目录指向 storage/reports/
│           └── markdown.py
│
├── tests/                          # 测试套件 (与 src/ 1:1 映射)
│   ├── api/
│   │   ├── __init__.py
│   │   ├── test_llm_key_status.py
│   │   ├── test_mask_key.py
│   │   ├── test_messages.py
│   │   ├── test_report.py
│   │   └── routes/
│   │       ├── conftest.py
│   │       ├── test_chat_route.py
│   │       └── test_llm_status.py
│   ├── cli/
│   │   └── ket_partner/
│   │       ├── __init__.py
│   │       ├── test_chat_logger.py
│   │       ├── test_commands.py
│   │       └── test_main.py
│   ├── flow/
│   │   └── ket_partner/
│   │       ├── conftest.py
│   │       ├── test_config.py
│   │       ├── test_graph.py
│   │       ├── test_persistence_protocol.py
│   │       ├── test_state.py
│   │       ├── test_sentence_domain.py # ★ NEW(合并): 对应 sentence_domain.py 单元测试
│   │       ├── test_vocab_domain.py    # ★ NEW(合并): 对应 vocab_domain.py 单元测试
│   │       └── test_dialogue_domain.py # ★ NEW(合并): 对应 dialogue_domain.py 单元测试
│   ├── integration/
│   │   ├── conftest.py
│   │   ├── test_chat_real_llm.py
│   │   └── test_graph_integration.py
│   ├── persistence/
│   │   ├── __init__.py
│   │   ├── conftest.py
│   │   ├── test_bootstrap.py
│   │   ├── test_migration.py
│   │   ├── test_models.py
│   │   ├── test_repos.py
│   │   └── test_schema.py
│   └── reporting/
│       ├── __init__.py
│       ├── conftest.py
│       └── ket_partner/
│           ├── __init__.py
│           ├── test_categories.py
│           ├── test_exporter.py
│           └── test_markdown.py
│
└── web/                            # 前端 Web 应用 (Vue 3 + Vite)
    ├── package.json / vite.config.ts / index.html
    └── src/
        ├── App.vue / main.ts / router.ts
        ├── api/ / components/ / stores/ / views/
```

---

## 三、旧文件合并与清理清单

### 1. `src/flow/ket_partner/` 微型文件归并映射表

为解决 CCP 碎片化问题，将 12 个微型文件按业务领域因果关系合并为 3 个高内聚领域模块：

| 原旧文件路径 | 动作 | 合并目标模块 | 包含的主要逻辑 |
| :--- | :---: | :--- | :--- |
| `sentence_generator.py` | 🔀 合并 | `sentence_domain.py` | 结合词汇/语义生成提示与目标句产出 |
| `sentence_validator.py` | 🔀 合并 | `sentence_domain.py` | 语法正确性与目标词包含性校验 |
| `sentence_naturalness.py` | 🔀 合并 | `sentence_domain.py` | 地道表达与自然度评级 |
| `sentence_orchestration.py` | 🔀 合并 | `sentence_domain.py` | 句子生成多分支编排与重试逻辑 |
| `multi_word_target.py` | 🔀 合并 | `sentence_domain.py` | 多词目标短语匹配补丁逻辑 |
| `vocab_selector.py` | 🔀 合并 | `vocab_domain.py` | 结合掌握度与历史记录的目标词选择 |
| `word_meaning_lookup.py` | 🔀 合并 | `vocab_domain.py` | 单词/短语中文释义与例句检索 |
| `mastery.py` | 🔀 合并 | `vocab_domain.py` | 掌握度算法计算与更新生成 |
| `input_classifier.py` | 🔀 合并 | `dialogue_domain.py` | 用户输入意图分类（答题/提问/闲聊） |
| `profile_summarizer.py` | 🔀 合并 | `dialogue_domain.py` | 用户对话特征与弱项总结 |
| `translation_evaluator.py` | 🔀 合并 | `dialogue_domain.py` | 用户翻译打分与纠错反馈 |
| `output_format.py` | 🔀 合并 | `dialogue_domain.py` | 节点输出文本格式化与排版拼接 |

*说明：合并完成后，上述 12 个微型 `.py` 文件将从代码库中物理删除。*

### 2. 测试文件合并映射表

| 原测试文件 | 动作 | 目标测试模块 |
| :--- | :---: | :--- |
| `test_sentence_generator.py`<br>`test_sentence_validator.py`<br>`test_sentence_naturalness.py`<br>`test_sentence_orchestration.py`<br>`test_multi_word_target.py` | 🔀 合并 | `tests/flow/ket_partner/test_sentence_domain.py` |
| `test_vocab_selector.py`<br>`test_word_meaning_lookup.py`<br>`test_mastery.py` | 🔀 合并 | `tests/flow/ket_partner/test_vocab_domain.py` |
| `test_input_classifier.py`<br>`test_profile_summarizer.py`<br>`test_translation_evaluator.py`<br>`test_output_format.py` | 🔀 合并 | `tests/flow/ket_partner/test_dialogue_domain.py` |

### 3. 冗余与无用数据文件清理清单

1. **删除冗余 DB 文件**：
   - 彻底删除 `src/ket_partner.db`。
   - 移动根目录 `ket_partner.db` 到 `storage/db/ket_partner.db`。
2. **清理报告文件**：
   - 将 `src/learning_report_*.md` 移动归档至 `storage/reports/`。
3. **保持 `src/flow/common.py` 现状**：
   - `src/flow/common.py` 先前已清理完成，仅保留核心 `llm_max` 与 `llm_flash` 配置。

---

## 四、存储隔离与配置规范

1. **`.gitignore` 规则**：
   ```gitignore
   # Dynamic Storage Directory
   storage/db/*.db
   storage/reports/*.md
   storage/logs/*.log
   !storage/**/.gitkeep
   ```
2. **路径配置收拢**：
   - `src/api/settings.py` 中的 `DB_PATH` 默认值修正为 `Path(__file__).resolve().parents[2] / "storage" / "db" / "ket_partner.db"`（推荐使用绝对路径或相对项目根路径解析）。
   - `src/cli/ket_partner/main.py` 中的 `DEFAULT_DB` 指向 `storage/db/ket_partner.db`。
   - `src/reporting/ket_partner/exporter.py` 默认输出路径指向 `storage/reports/`。

---

## 五、模块依赖架构约束（DAG 单向无环）

依照 CLAUDE.md §10：

```
       ┌──────────────────────────────────────┐
       │       src/flow/ket_partner/          │  ← 纯 Agent 领域逻辑
       │   graph.py / nodes.py / agent.py /   │     (零 runtime 依赖外部 persistence)
       │   sentence/vocab/dialogue_domain.py  │
       └──────────────────┬───────────────────┘
                          │ TYPE_CHECKING
                          ▼
             ┌─────────────────────────┐
             │    src/persistence/     │  ← 独立数据持久化层
             └────────────┬────────────┘
                          │
                          ▼
       ┌──────────────────────────────────────┐
       │   src/api/  &  src/cli/ket_partner/  │  ← Composition Root 组装入口
       └──────────────────────────────────────┘
```

---

## 六、静态检查与验证要求

在实施重构后，必须依次运行以下命令并确保全部无错误输出：

1. `ruff check .` （确保零语法/代码风格/未定义变量错误）
2. `mypy src/` （严格类型检查 100% 通过）
3. `pytest` （所有单元测试与集成测试全部通过）
