# 项目文件结构调整与代码抗碎片化 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 根据 [2026-07-30-file-structure-and-consolidation-design.md](file:///D:/Workspace/HBuilderProjects/%E8%8B%B1%E8%AF%ADKET/%E8%8B%B1%E8%AF%AD/docs/superpowers/specs/2026-07-30-file-structure-and-consolidation-design.md) 重新组织项目文件结构，建立 `storage/` 动态存储目录、消除 `src/flow/ket_partner/` 中的 12 个微型碎片文件（合并为 3 大高内聚领域模块），并按 1:1 映射调整 `tests/` 目录。

**Architecture:** 
1. 采用统一 `storage/` 目录（`storage/db/`, `storage/reports/`, `storage/logs/`）存放运行期产物，在 `.gitignore` 中彻底隔离。
2. 遵循 CLAUDE.md §4.6 (CCP 共同封闭原则) 与 §10.1 (LangGraph 三层架构)，将原本散落的 12 个 LLM 微型模块合并为 `sentence_domain.py`、`vocab_domain.py` 与 `dialogue_domain.py`，并将 Graph Node 定义抽取至 `nodes.py`。
3. 同步重构 `tests/flow/ket_partner/` 下的微型测试文件为 1:1 的领域测试模块。

**Tech Stack:** Python 3.12+, LangGraph, FastAPI, SQLite, Pytest, Ruff, Mypy

## Global Constraints

- **Python**: 3.12+
- **Code Standards**: CLAUDE.md (Zero bare `except Exception`, explicit types, `mypy` strict passing, `ruff` passing).
- **DAG Dependency**: `flow/ket_partner` has zero runtime dependency on `persistence`, `cli`, or `api`.
- **Database Location**: Default SQLite database at `storage/db/ket_partner.db`.

---

### Task 1: 创设 `storage/` 隔离存储与配置更新

**Files:**
- Create: `storage/db/.gitkeep`, `storage/reports/.gitkeep`, `storage/logs/.gitkeep`
- Modify: `.gitignore`, `pytest.ini`

**Interfaces:**
- Consumes: N/A
- Produces: `storage/` directory layout for DB, reports, and logs.

- [ ] **Step 1: 创建 `storage/` 子目录及其 `.gitkeep` 占位符**

```bash
mkdir -p storage/db storage/reports storage/logs
touch storage/db/.gitkeep storage/reports/.gitkeep storage/logs/.gitkeep
```

- [ ] **Step 2: 在 `.gitignore` 中增加 `storage/` 隔离规则**

在 `.gitignore` 文件末尾添加：
```gitignore
# Dynamic Storage Directory
storage/db/*.db
storage/reports/*.md
storage/logs/*.log
!storage/**/.gitkeep
```

- [ ] **Step 3: 更新 `pytest.ini` 配置**

确保 `pytest.ini` 中包含 warning 严格拦截规则：
```ini
filterwarnings =
    error::pytest.PytestUnknownMarkWarning
```

- [ ] **Step 4: 运行 pytest 验证配置加载**

Run: `pytest`
Expected: PASS

- [ ] **Step 5: Commit Task 1**

```bash
git add .gitignore pytest.ini storage/
git commit -m "chore: set up storage runtime directory and update gitignore"
```

---

### Task 2: 重构后端 `src/flow/ket_partner/` 高内聚领域模块与节点层

**Files:**
- Create: `src/flow/ket_partner/sentence_domain.py`, `src/flow/ket_partner/vocab_domain.py`, `src/flow/ket_partner/dialogue_domain.py`, `src/flow/ket_partner/nodes.py`
- Modify: `src/flow/ket_partner/graph.py`, `src/flow/ket_partner/agent.py`
- Delete: `src/flow/ket_partner/sentence_generator.py`, `src/flow/ket_partner/sentence_validator.py`, `src/flow/ket_partner/sentence_naturalness.py`, `src/flow/ket_partner/sentence_orchestration.py`, `src/flow/ket_partner/multi_word_target.py`, `src/flow/ket_partner/vocab_selector.py`, `src/flow/ket_partner/word_meaning_lookup.py`, `src/flow/ket_partner/mastery.py`, `src/flow/ket_partner/input_classifier.py`, `src/flow/ket_partner/profile_summarizer.py`, `src/flow/ket_partner/translation_evaluator.py`, `src/flow/ket_partner/output_format.py`

**Interfaces:**
- Consumes: `AgentState`, `KetConfig`, `KETPartnerRepos`
- Produces: 
  - `sentence_domain.py`: `generate_with_fallback`, `apply_multiword_target_patch`, `validate_sentence`, `check_naturalness`
  - `vocab_domain.py`: `select_vocab_words`, `lookup_word_meanings`, `apply_mastery_updates`
  - `dialogue_domain.py`: `classify_user_input`, `summarize_user_profile`, `evaluate_translation`, `format_output_text`
  - `nodes.py`: Concentrated node functions for LangGraph

- [ ] **Step 1: 创建 `sentence_domain.py`**

将造句、语法校验、自然度评估、句子编排与多词目标匹配合并至 `src/flow/ket_partner/sentence_domain.py`，暴露对外干净接口。

- [ ] **Step 2: 创建 `vocab_domain.py`**

将选词、查释义、掌握度计算与更新逻辑合并至 `src/flow/ket_partner/vocab_domain.py`。

- [ ] **Step 3: 创建 `dialogue_domain.py`**

将输入分类、用户画像总结、翻译评估与输出格式化逻辑合并至 `src/flow/ket_partner/dialogue_domain.py`。

- [ ] **Step 4: 创建 `nodes.py` 并重构 `graph.py` 与 `agent.py`**

在 `nodes.py` 中集中定义 LangGraph 节点函数，更新 `graph.py`（边编排与条件路由）和 `agent.py`（`KETPartnerAgent` 类）引用新的领域模块。

- [ ] **Step 5: 删除 12 个微型旧文件**

```bash
rm src/flow/ket_partner/sentence_generator.py \
   src/flow/ket_partner/sentence_validator.py \
   src/flow/ket_partner/sentence_naturalness.py \
   src/flow/ket_partner/sentence_orchestration.py \
   src/flow/ket_partner/multi_word_target.py \
   src/flow/ket_partner/vocab_selector.py \
   src/flow/ket_partner/word_meaning_lookup.py \
   src/flow/ket_partner/mastery.py \
   src/flow/ket_partner/input_classifier.py \
   src/flow/ket_partner/profile_summarizer.py \
   src/flow/ket_partner/translation_evaluator.py \
   src/flow/ket_partner/output_format.py
```

- [ ] **Step 6: 验证 ruff 与 mypy**

Run: `ruff check src/flow/ket_partner/ && mypy src/flow/ket_partner/`
Expected: PASS with 0 errors

- [ ] **Step 7: Commit Task 2**

```bash
git add src/flow/ket_partner/
git commit -m "refactor(flow/ket_partner): consolidate micro-modules into high-cohesion domain modules per CCP"
```

---

### Task 3: 默认路径收拢至 `storage/` 与无用文件清理

**Files:**
- Modify: `src/api/settings.py`, `src/cli/ket_partner/main.py`, `src/cli/ket_partner/chat_logger.py`, `src/persistence/bootstrap.py`, `src/reporting/ket_partner/exporter.py`
- Move/Delete: `src/ket_partner.db` (delete), `ket_partner.db` -> `storage/db/ket_partner.db`, `src/learning_report_*.md` -> `storage/reports/`

**Interfaces:**
- Consumes: `Path.resolve()` for absolute path pointing to `storage/`
- Produces: Standardized storage paths across persistence, api, cli, and reporting layers.

- [ ] **Step 1: 修改 `src/api/settings.py`**

将 `DB_PATH` 修正为相对于根目录的 `storage/db/ket_partner.db`：
```python
DB_PATH: str = str(Path(__file__).resolve().parents[2] / "storage" / "db" / "ket_partner.db")
```

- [ ] **Step 2: 修改 `src/cli/ket_partner/main.py` 与 `chat_logger.py`**

将 `DEFAULT_DB` 设置为 `storage/db/ket_partner.db`，日志写入 `storage/logs/`。

- [ ] **Step 3: 修改 `src/persistence/bootstrap.py` 与 `src/reporting/ket_partner/exporter.py`**

将默认数据库和导出报告文件路径统一修正至 `storage/db/` 和 `storage/reports/`。

- [ ] **Step 4: 迁移/删除杂项文件**

```bash
rm -f src/ket_partner.db
mv src/learning_report_*.md storage/reports/ 2>/dev/null || true
if [ -f ket_partner.db ]; then mv ket_partner.db storage/db/ket_partner.db; fi
```

- [ ] **Step 5: 验证路径加载**

Run: `ruff check src/ && mypy src/`
Expected: PASS with 0 errors

- [ ] **Step 6: Commit Task 3**

```bash
git add src/ storage/
git commit -m "refactor(storage): update default DB, report and log paths to storage/"
```

---

### Task 4: 合并测试套件至 1:1 领域测试模块

**Files:**
- Create: `tests/flow/ket_partner/test_sentence_domain.py`, `tests/flow/ket_partner/test_vocab_domain.py`, `tests/flow/ket_partner/test_dialogue_domain.py`
- Modify: `tests/flow/ket_partner/test_graph.py`
- Delete: `tests/flow/ket_partner/test_sentence_generator.py`, `tests/flow/ket_partner/test_sentence_validator.py`, `tests/flow/ket_partner/test_sentence_naturalness.py`, `tests/flow/ket_partner/test_sentence_orchestration.py`, `tests/flow/ket_partner/test_multi_word_target.py`, `tests/flow/ket_partner/test_vocab_selector.py`, `tests/flow/ket_partner/test_word_meaning_lookup.py`, `tests/flow/ket_partner/test_mastery.py`, `tests/flow/ket_partner/test_input_classifier.py`, `tests/flow/ket_partner/test_profile_summarizer.py`, `tests/flow/ket_partner/test_translation_evaluator.py`, `tests/flow/ket_partner/test_output_format.py`

**Interfaces:**
- Consumes: Merged functions in `sentence_domain.py`, `vocab_domain.py`, `dialogue_domain.py`
- Produces: 1:1 unit test files mirroring domain modules.

- [ ] **Step 1: 创建 `test_sentence_domain.py`**

合并造句、校验、自然度评估与编排测试，更新被测函数的 import 来源为 `sentence_domain`。

- [ ] **Step 2: 创建 `test_vocab_domain.py`**

合并选词、释义与掌握度计算测试，更新 import 来源为 `vocab_domain`。

- [ ] **Step 3: 创建 `test_dialogue_domain.py`**

合并输入分类、画像总结、翻译评估与格式化测试，更新 import 来源为 `dialogue_domain`。

- [ ] **Step 4: 删除旧的 12 个微型测试文件**

```bash
rm tests/flow/ket_partner/test_sentence_generator.py \
   tests/flow/ket_partner/test_sentence_validator.py \
   tests/flow/ket_partner/test_sentence_naturalness.py \
   tests/flow/ket_partner/test_sentence_orchestration.py \
   tests/flow/ket_partner/test_multi_word_target.py \
   tests/flow/ket_partner/test_vocab_selector.py \
   tests/flow/ket_partner/test_word_meaning_lookup.py \
   tests/flow/ket_partner/test_mastery.py \
   tests/flow/ket_partner/test_input_classifier.py \
   tests/flow/ket_partner/test_profile_summarizer.py \
   tests/flow/ket_partner/test_translation_evaluator.py \
   tests/flow/ket_partner/test_output_format.py
```

- [ ] **Step 5: 运行全量 pytest 测试**

Run: `pytest`
Expected: ALL PASS

- [ ] **Step 6: Commit Task 4**

```bash
git add tests/flow/ket_partner/
git commit -m "test(flow/ket_partner): consolidate micro-tests into 1:1 domain test modules"
```

---

### Task 5: 静态检查终验与全项目结构校验

**Files:**
- Audit: Entire repository structure and code quality

- [ ] **Step 1: 运行 ruff 语法与代码风格全量检查**

Run: `ruff check .`
Expected: PASS with 0 warnings/errors

- [ ] **Step 2: 运行 mypy 严格类型全量检查**

Run: `mypy src/`
Expected: PASS with 0 type errors

- [ ] **Step 3: 运行 pytest 全套测试集合**

Run: `pytest`
Expected: ALL PASS

- [ ] **Step 4: Commit Final State**

```bash
git add .
git commit -m "chore: final verification of file structure reorganization and anti-fragmentation"
```
