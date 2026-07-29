# KET Partner 包重构设计

**日期**：2026-07-29
**分支**：refactor/code-compliance
**状态**：spec 待审阅

---

## 一、背景与动机

`src/flow/ket_partner/` 当前混合了 4 类职责：agent 业务编排 / 持久化 / CLI 入口 / 报告导出。具体症状：

- `db.py` (762 行)：schema DDL + 5 个 Repo 类 + `Repos` 聚合 + 迁移 + CSV 导入 + `init_db` 工厂 + 业务规则（`_derive_status` / `MASTERY_CAP`）+ 值对象 `WordRef` —— 持久化所有层面揉一个文件
- `agent.py` (552 行)：13 个 LangGraph 节点方法 + `_generate_with_fallback` (110+ 行业务编排) + `_validate_and_categorize` + graph 装配 `compile()` + `build_agent` / `autonomous` 工厂 —— 违反 CLAUDE.md §10.1「节点定义散落在 agent.py 类方法里」
- `nodes.py` (99 行)：装的其实是被节点调用的业务 helper（`apply_mastery_updates` / `format_output_text`），不是节点 —— 文件名与内容错位
- `commands.py::_export_stats`：自己 `init_db` + 访问 `repos._db.close()` 私有属性 —— 启动装配散落在 CLI 命令里
- `exporter.py::_fetch_all_stats`：直接 `repos.stats._db.execute(...)` —— 绕过 Repo 抽象
- `flow/agent.py` (顶层 Autonomous/Consensus)：dead code，全仓库无 import
- `flow/common.py`：4 块 dead code（`IS_RUNNING_IN_PYTEST` / `llm_plus` / `llm_doubao` / `doubao_api`）+ `llm_flash` 的 `extra_body` 与同类常量不一致

用户诉求：agent 应是「纯粹的 agent」，persistence / CLI / reporting 各自独立；从整个项目角度重新组织代码，为多用户未来留位置。

---

## 二、目标架构总览

### 三层依赖方向

```
        ┌─────────────────────────────────────┐
        │       src/flow/ket_partner/         │  ← 域核心 (PURE)
        │   agent / Protocol / state /        │
        │   config / domain LLM modules       │
        └────┬──────────────────────────┬─────┘
             │ TYPE_CHECKING            │
             │   (WordRef, 仅类型层)    │ runtime
             ▼                          │ (KetConfig / KETPartnerRepos Protocol)
    ┌──────────────────┐                ▼
    │  persistence/    │       ┌──────────────────┐
    │  (Repos 实现)    │       │  reporting/      │
    │                  │       │  ket_partner/    │
    └────┬─────────────┘       └────┬─────────────┘
         │ runtime                  │ runtime
         │ (init_db / Repos)        │ (exporter)
         ▼                          ▼
    ┌────────────────────────────────────────────┐
    │  src/api/  +  src/cli/ket_partner/         │
    │       (composition root / 入口层)          │
    └────────────────────────────────────────────┘
```

**关键不变量**：
- `flow/ket_partner/` 对 `persistence/` / `reporting/` / `cli/` / `api/` 零运行时依赖；仅 `TYPE_CHECKING` 引用 `persistence.models.WordRef`。
- `persistence/` 对 `flow/ket_partner/` / `reporting/` / `cli/` / `api/` 零依赖（运行时与类型层皆然）—— Option B 删除分类 SQL 方法后，persistence 不再需要 `KetConfig`。
- `reporting/` 仅依赖 `flow/ket_partner/`（runtime：`KetConfig` + `KETPartnerRepos` Protocol）；不依赖 persistence / cli / api。
- `api/` + `cli/` 是 composition root，装配一切。

### 顶层包布局

```
src/
├── api/                            # 现有 web 层（仅 import 路径与 /report 路由内部改造）
├── flow/
│   ├── common.py                   # 清理 dead code
│   ├── agent.py                    # ★ 删除（dead code）
│   └── ket_partner/                # ★ PURE agent 包
│       ├── agent.py                # KETPartnerAgent 类（仅节点方法 + __init__ + aclose）
│       ├── graph.py                # build_agent + wire_graph + 路由函数 + passthrough
│       ├── persistence.py          # ★ NEW: KETPartnerRepos Protocol + 5 sub-Protocol + get_repos
│       ├── sentence_orchestration.py   # ★ NEW: 句子生成编排
│       ├── mastery.py              # 改名自 nodes.py
│       ├── output_format.py        # 改名自 nodes.py
│       ├── state.py / config.py    # 不动
│       └── (9 个 LLM 领域模块)      # 不动
├── persistence/                    # ★ NEW 项目级持久化
│   ├── schema.py                   # DDL
│   ├── models.py                   # WordRef + MASTERY_CAP + derive_status
│   ├── repos.py                    # 5 Repo + Repos
│   ├── bootstrap.py                # init_db + _import_csv
│   └── migration.py                # migrate_old_schema_if_needed
├── cli/ket_partner/                # ★ NEW 项目级 CLI
│   ├── main.py
│   ├── commands.py
│   └── chat_logger.py
└── reporting/ket_partner/          # ★ NEW 项目级报告
    ├── exporter.py
    ├── categories.py               # ★ 分类规则单一来源（Option B）
    └── markdown.py
```

---

## 三、`src/persistence/` 详细规范

### `schema.py`

**职责**：单一常量 `SCHEMA_SQL`，纯 DDL，无逻辑。

```python
SCHEMA_SQL: str
    """7 张表的 CREATE TABLE IF NOT EXISTS + 索引 DDL：
    ket_vocabulary, ket_vocab_topics, vocab_stats, conversation_log,
    kid_profile, users, recent_sentences
    """
```

### `models.py`

**职责**：跨 repo 共享的值对象 + 业务常量 + 状态派生纯函数。

```python
from typing import NamedTuple

class WordRef(NamedTuple):
    """(word, context) 对 —— 练习单位。贯穿 vocab_selector / agent / evaluator。
    字段：
      word: 单词规范形式
      context: 语境（空字符串表示默认义项）
    """
    word: str
    context: str = ""

MASTERY_CAP: int  # = 2，mastery 封顶值，控制已掌握词的降级路径

def derive_status(
    current_status: str | None,
    mastery_score: int,
    is_target: bool = False,
) -> str:
    """根据当前 status + mastery_score + 是否为 target 词派生新 status。
    返回值之一：'mastered' | 'learning' | 'exposed' | current_status（保持）

    规则：
      - mastery_score >= MASTERY_CAP          → 'mastered'
      - current_status == 'mastered' 且降级    → 'learning' if score <= 1 else 'mastered'
      - is_target                              → 'learning'
      - current_status is None                 → 'exposed'
      - 否则                                   → current_status

    重命名自 db.py::_derive_status（去下划线，跨模块公开）。
    """
```

### `repos.py`

**职责**：5 个 per-user Repo 类 + `Repos` 聚合门面。每个 Repo 暴露 narrow interface，封装 SQL 细节。

```python
class VocabRepo:
    """ket_vocabulary / ket_vocab_topics 表的读访问。"""

    def __init__(self, db: aiosqlite.Connection, user_id: str = "default") -> None: ...

    async def get_topics_for_word(self, word: str, context: str = "") -> list[str]:
        """查词的所有 topic，按字母序排序。"""

    async def get_ket_word(self, word: str, context: str = "") -> WordRef | None:
        """精确 (word, context) 查询；大小写不敏感。"""

    async def get_ket_word_any_context(self, word: str) -> WordRef | None:
        """跨 context 查词；优先返回 context='' 的默认义项。"""

    async def words_in_topic_without_stats(self, topic: str) -> list[WordRef]:
        """topic 下尚未有 stats 记录的词，随机取 1 个。"""

    async def unexposed_notopic_words(self) -> list[WordRef]:
        """无 topic 且无 stats 的词，随机取 1 个（兜底用）。"""

    async def topics_with_unmastered(self, exclude: str | None = None) -> list[str]:
        """仍有未掌握词的 topic，随机取 1 个；可排除指定 topic。"""

    async def total_count(self) -> int:
        """ket_vocabulary 总词数。"""


class StatsRepo:
    """vocab_stats 表的读写 + mastery 派生。

    Option B 删除 `_category_where_sql` / `count_by_category` / `list_by_category`
    后，本类不再需要 KetConfig —— 分类规则全部走 reporting/ket_partner/categories.py。
    """

    def __init__(
        self,
        db: aiosqlite.Connection,
        user_id: str = "default",
    ) -> None: ...

    async def get(self, word: str, context: str = "") -> dict | None:
        """取单条 stats row；返回 dict 含 exposed/correct/wrong_count/mastery_score/status 等。"""

    async def apply_delta(
        self,
        word: str,
        context: str = "",
        delta: int = 0,
        exposed: bool = False,
        is_target: bool = False,
    ) -> dict | None:
        """对 (word, context) 应用增量；不存在则插入。
        - delta > 0 → correct_count +1
        - delta < 0 → wrong_count +1
        - exposed=True → exposed_count +1
        返回更新后的 stats dict；context='' 且 vocab 无默认义项时返回 None。
        """

    async def learning_count(self) -> int:
        """status='learning' 的词数（用于 refill 水位判断）。"""

    async def oldest_learning_word(self) -> WordRef | None:
        """status='learning' 中 last_seen_at 最早的词；fallback 到 'exposed'。"""

    async def increment_exposed(
        self, word: str, context: str = "", is_target: bool = False,
    ) -> None:
        """便捷方法：仅增 exposed_count，不改 delta。"""

    async def list_all_with_vocab(self) -> list[dict]:
        """★ NEW：vocab_stats LEFT JOIN ket_vocabulary 全量返回（含未练过的词）。
        替代 exporter.py 原对 repos.stats._db.execute 的私有访问。
        每行 dict 含 word/context/pos/exposed_count/correct_count/wrong_count/mastery_score/status。
        """


class ProfileRepo:
    """kid_profile + users 表的读写。"""

    def __init__(self, db: aiosqlite.Connection, user_id: str = "default") -> None: ...

    async def get(self) -> dict:
        """联表查 user + profile；user_id 不存在时返回默认值（nickname='宝贝', age=8 等）。"""

    async def update(self, **fields) -> None:
        """字段级更新；weakness_words 自动 json 序列化。
        允许字段集合：profile_allowed = {total_turns, weakness_words, dialogue_strategy,
        in_refill_mode, last_new_word_turn, last_summary_turn, current_topic}；
        user_allowed = {nickname, age}。
        """


class LogRepo:
    """conversation_log 表的读写。"""

    def __init__(self, db: aiosqlite.Connection, user_id: str = "default") -> None: ...

    async def append(
        self,
        role: str,
        content: str,
        words_used: list[str] | None = None,
        target_words: list[dict[str, str]] | None = None,
        turn_id: int | None = None,
    ) -> None:
        """追加一条对话日志；words_used/target_words 自动 json 序列化。"""

    async def recent(self, limit: int = 5) -> list[dict]:
        """取最近 N 条日志，按 id 倒序查再 reverse 回时间正序。"""

    async def append_session_start(self) -> None:
        """便捷方法：写入 role='system', content='session_start' 标记。"""

    async def last_ai_message(self) -> dict | None:
        """查本 session（最近 session_start 之后）的最后一条 AI 消息。
        返回 dict 含 content/words_used/target_words；无则 None。
        """


class RecentSentencesRepo:
    """recent_sentences 表的读写。"""

    def __init__(self, db: aiosqlite.Connection, user_id: str = "default") -> None: ...

    async def list_recent(self, limit: int = 20) -> list[str]:
        """取最近 N 条英文句子（去重前），按时间倒序。"""

    async def append(self, sentence: str, window: int = 20) -> None:
        """追加一条；同时修剪到保留最近 window 条。"""

    async def list_recent_scaffolding(self, window: int = 20) -> list[list[str]]:
        """取最近 window 句的 token 化结果（小写），供句子生成时回避重复用词。"""


class Repos:
    """5 个 per-user Repo 的门面。每请求构造一次，user_id 隔离。"""

    def __init__(
        self,
        db: aiosqlite.Connection,
        user_id: str = "default",
    ) -> None:
        """构造时同时实例化 vocab/stats/profile/log/recent 5 个 Repo。"""

    @classmethod
    def for_user(
        cls,
        db: aiosqlite.Connection,
        user_id: str = "default",
    ) -> "Repos":
        """语义同 __init__，显式命名提升可读性。"""

    async def close(self) -> None:
        """关闭底层 db 连接。"""

    # 类级属性注解（运行时由 __init__ 赋值）：
    vocab: VocabRepo
    stats: StatsRepo
    profile: ProfileRepo
    log: LogRepo
    recent: RecentSentencesRepo
```

**Option B 删除**（原 StatsRepo 方法，本次消除）：
- `_category_where_sql(category) -> tuple[str, tuple]`
- `count_by_category(category) -> int`
- `list_by_category(category, offset=0, limit=100) -> list[dict]`

### `bootstrap.py`

**职责**：DB 连接初始化 + CSV 导入。composition root 调用。

```python
async def init_db(
    db_path: str,
    csv_path: str | None = None,
    default_nickname: str = "宝贝",
    default_age: int = 8,
) -> aiosqlite.Connection:
    """打开 DB 连接，设置 PRAGMA (WAL / busy_timeout)，运行 migration，
    executescript(SCHEMA_SQL)，seed default user 与 kid_profile，
    可选 CSV 导入。返回活跃连接。
    """

async def _import_csv(db: aiosqlite.Connection, csv_path: str) -> None:
    """从 CSV 导入 KET 词汇到 ket_vocabulary + ket_vocab_topics。
    期望列：word / part_of_speech / topic（分号分隔）/ context。
    保持私有 —— 仅 bootstrap.py 内部调用。
    """
```

### `migration.py`

**职责**：旧版单用户 schema 升级到多租户。

```python
async def migrate_old_schema_if_needed(db: aiosqlite.Connection) -> None:
    """检测并升级旧版 CLI 数据库：
    - vocab_stats 缺 user_id 列 → ADD COLUMN
    - conversation_log 缺 user_id 列 → ADD COLUMN
    - kid_profile 为旧结构（id 列、无 user_id）→ 创建 users 表、DROP + 重建 kid_profile
    无 schema 时 no-op。
    重命名自 db.py::_migrate_old_schema_if_needed（去下划线，跨模块公开）。
    """
```

### `__init__.py`

仅再导出外部使用的：
```python
from persistence.bootstrap import init_db
from persistence.models import WordRef, MASTERY_CAP, derive_status
from persistence.repos import (
    VocabRepo, StatsRepo, ProfileRepo, LogRepo, RecentSentencesRepo, Repos,
)
```

`SCHEMA_SQL` / `_import_csv` / `migrate_old_schema_if_needed` 不再导出。

**关键不变量**：`persistence/` 包内任何文件**禁止** `from flow.ket_partner...` import。Option B 删除分类 SQL 方法后，`StatsRepo` / `Repos` 都不再需要 `KetConfig`，persistence 对域配置的运行时依赖归零。

---

## 四、`src/flow/ket_partner/` 详细规范

### `persistence.py` (NEW)

**职责**：agent 对持久化的契约。agent 包零运行时依赖 persistence，仅 TYPE_CHECKING 引用 `WordRef`。

```python
from __future__ import annotations
from typing import TYPE_CHECKING, Protocol, runtime_checkable
from langchain_core.runnables import RunnableConfig

if TYPE_CHECKING:
    from persistence.models import WordRef  # 仅 type-check


class VocabRepoProtocol(Protocol):
    async def get_topics_for_word(self, word: str, context: str = "") -> list[str]: ...
    async def get_ket_word(self, word: str, context: str = "") -> WordRef | None: ...
    async def get_ket_word_any_context(self, word: str) -> WordRef | None: ...
    async def words_in_topic_without_stats(self, topic: str) -> list[WordRef]: ...
    async def unexposed_notopic_words(self) -> list[WordRef]: ...
    async def topics_with_unmastered(self, exclude: str | None = None) -> list[str]: ...
    async def total_count(self) -> int: ...


class StatsRepoProtocol(Protocol):
    async def get(self, word: str, context: str = "") -> dict | None: ...
    async def apply_delta(
        self, word: str, context: str = "", delta: int = 0,
        exposed: bool = False, is_target: bool = False,
    ) -> dict | None: ...
    async def learning_count(self) -> int: ...
    async def oldest_learning_word(self) -> WordRef | None: ...
    async def increment_exposed(
        self, word: str, context: str = "", is_target: bool = False,
    ) -> None: ...
    async def list_all_with_vocab(self) -> list[dict]: ...


class ProfileRepoProtocol(Protocol):
    async def get(self) -> dict: ...
    async def update(self, **fields) -> None: ...


class LogRepoProtocol(Protocol):
    async def append(
        self, role: str, content: str,
        words_used: list[str] | None = None,
        target_words: list[dict[str, str]] | None = None,
        turn_id: int | None = None,
    ) -> None: ...
    async def recent(self, limit: int = 5) -> list[dict]: ...
    async def append_session_start(self) -> None: ...
    async def last_ai_message(self) -> dict | None: ...


class RecentSentencesRepoProtocol(Protocol):
    async def list_recent(self, limit: int = 20) -> list[str]: ...
    async def append(self, sentence: str, window: int = 20) -> None: ...
    async def list_recent_scaffolding(self, window: int = 20) -> list[list[str]]: ...


@runtime_checkable
class KETPartnerRepos(Protocol):
    """Agent 对持久化的契约。具体实现在 persistence/repos.py::Repos。

    @runtime_checkable 让 isinstance(Repos(...), KETPartnerRepos) 可用，
    供 test_persistence_protocol.py 验证契约一致性（仅检查属性存在，不查方法签名）。

    Single-Writer / 调用方约束：
    - vocab: 仅 VocabRepo / select_target_word / evaluate_translation_node 读
    - stats: 仅 StatsRepo / apply_mastery_updates / generate_sentence_node / _run_summary_safe 读写
    - profile: 仅 ProfileRepo / 多节点读，select_target_word_node + persist_turn_node + _run_summary_safe 写
    - log: 仅 LogRepo / init_state + persist_turn_node 写，init_state 读
    - recent: 仅 RecentSentencesRepo / generate_sentence_node 写 + 读
    """
    vocab: VocabRepoProtocol
    stats: StatsRepoProtocol
    profile: ProfileRepoProtocol
    log: LogRepoProtocol
    recent: RecentSentencesRepoProtocol


def get_repos(config: RunnableConfig) -> KETPartnerRepos:
    """集中 LangGraph config['configurable']['repos'] 访问点。
    节点方法唯一允许的 repos 获取方式。
    """
```

### `agent.py` (瘦身)

**职责**：仅装 `KETPartnerAgent` 类 —— 节点方法 + agent 级状态（LLM clients、KetConfig、bg task set）。

```python
class KETPartnerAgent:
    def __init__(
        self,
        llm_flash: BaseChatModel,
        llm_smart: BaseChatModel,
        config: KetConfig,
    ) -> None:
        """持有 agent 级配置与 bg task set。
        每请求状态（Repos / user_info）走 LangGraph config 流转，不缓存到 self。
        """

    # === 13 节点方法（签名不变；内部访问 repos 改走 get_repos(config)）===

    async def init_state(self, state: BTPKetState, config: RunnableConfig) -> dict:
        """读 profile + last_ai_message，初始化 topic / strategy / weakness /
        last_english_sentence / last_sentence_words / last_target_word 等字段。
        messages 超过 10 条时裁剪到最近 10 条。"""

    async def classify_intent_node(self, state: BTPKetState, config: RunnableConfig) -> dict:
        """调 classify_intent(llm_smart, last_english_sentence, kid_input)，
        返回 {intent, asked_word}。"""

    async def evaluate_translation_node(self, state: BTPKetState, config: RunnableConfig) -> dict:
        """调 evaluate_translation(llm_smart, ...) 评估翻译，过滤 wrong_words：
        - 丢 kid_translation == correct_translation 的项
        - 丢 correct_translation 空的项
        - 丢 word 不在 last_sentence_words 且不在 displayed_tokens 的项
        - 丢重复 key
        返回 {wrong_words, sentence_translation, overall_correct}。"""

    async def lookup_target_meaning_node(self, state: BTPKetState, config: RunnableConfig) -> dict:
        """idk 意图：调 lookup_sentence_translation(llm_flash, ...) 取整句翻译。
        返回 {sentence_translation}。"""

    async def lookup_asked_meaning_node(self, state: BTPKetState, config: RunnableConfig) -> dict:
        """asks_meaning 意图：调 lookup_word_meaning(llm_flash, ...) 取 queried word 释义。
        返回 {asked_word_meaning}。"""

    async def update_mastery_node(self, state: BTPKetState, config: RunnableConfig) -> dict:
        """委托 mastery.apply_mastery_updates(state, repos)。返回 {}。"""

    async def select_target_word_node(self, state: BTPKetState, config: RunnableConfig) -> dict:
        """调 vocab_selector.select_target_word(repos, profile, config) 选目标词。
        返回 {target_word, target_context}；无可用词时两者皆 None。"""

    async def generate_sentence_node(self, state: BTPKetState, config: RunnableConfig) -> dict:
        """★ 改造：调 sentence_orchestration.generate_with_fallback(...) 取
        (sentence, result, target, ctx)；apply_multiword_target_patch 后处理；
        repos.recent.append；逐词 increment_exposed；非 KET 词 lookup_word_meanings。
        返回 {last_sentence_words, last_english_sentence, _exposure_recorded,
        non_ket_annotations, [target_word], [target_context]}。"""

    async def format_output_node(self, state: BTPKetState, config: RunnableConfig) -> dict:
        """委托 output_format.format_output_text(state, sentence)，追加 AIMessage。
        返回 {messages: [...state.messages, AIMessage]}。"""

    async def explain_meaning_node(self, state: BTPKetState, config: RunnableConfig) -> dict:
        """asks_meaning 后续：渲染 '"asked" 的意思是「meaning」' + 继续提示。"""

    async def redirect_to_translate_node(self, state: BTPKetState, config: RunnableConfig) -> dict:
        """off_topic 后续：渲染'我们继续翻译练习吧'+ 上一句。"""

    async def compliance_redirect_node(self, state: BTPKetState, config: RunnableConfig) -> dict:
        """non_compliant 后续：渲染'换个健康话题'+ 上一句。"""

    async def persist_turn_node(self, state: BTPKetState, config: RunnableConfig) -> dict:
        """写对话日志（user + ai）；profile.total_turns +=1；
        若到达 summary.interval_turns 阈值，fire-and-forget _run_summary_safe。
        返回 {}。"""

    # === Background task 生命周期 ===

    async def _run_summary_safe(self, repos: KETPartnerRepos) -> None:
        """包装 run_profile_summary；捕获具体异常元组并 logger.warning，
        防止 background task 异常吞没。"""

    async def aclose(self, timeout: float = 2.0) -> None:
        """关闭时等待 bg tasks 完成；超时则 cancel 并 warning。"""
```

**删除（迁出）**：
- `compile()` → `graph.wire_graph()`
- `_generate_with_fallback()` → `sentence_orchestration.generate_with_fallback()`
- `_validate_and_categorize()` → `sentence_orchestration.validate_and_categorize()`
- `_route_call2()` → `graph.route_after_classify()`
- `_passthrough()` + `_route_after_init_state()` → `graph.passthrough_node()`
- 模块级 `build_agent()` → `graph.build_agent()`
- 模块级 `autonomous()` → **删除**（dead code）

### `graph.py` (扩充)

**职责**：图拓扑编排 + 路由函数 + 工厂。

```python
# === 路由函数（纯 state → str）===

def route_by_intent(state: BTPKetState) -> str:
    """format_output 后路由。intent ∈ {translation, idk} → 'select_target_word'；
    asks_meaning → 'explain_meaning'；off_topic → 'redirect_to_translate'；
    non_compliant → 'compliance_redirect'；fallback → 'select_target_word'。"""

def route_after_init(state: BTPKetState) -> str:
    """init_state 后路由。last_english_sentence is None → 'select_target_word'；
    否则 → 'classify_intent'。"""

def route_after_classify(state: BTPKetState) -> str:
    """classify_intent 后路由（改名自 KETPartnerAgent._route_call2）。
    translation → 'evaluate_translation'；idk → 'lookup_target_meaning'；
    asks_meaning → 'lookup_asked_meaning'；其他 → 'skip'。"""


# === Passthrough 节点（LangGraph 拓扑 hack）===

async def passthrough_node(state: BTPKetState, config: RunnableConfig) -> dict:
    """No-op 节点，仅为 add_conditional_edges 提供分支宿主。
    合并自原 KETPartnerAgent._passthrough 与 _route_after_init_state（两者皆 no-op）。
    返回 {}。"""


# === 图装配 ===

def wire_graph(builder: StateGraph, agent: KETPartnerAgent) -> None:
    """添加全部 13 节点 + 边到 StateGraph builder。
    提取自原 KETPartnerAgent.compile() body。包含：
    - 13 个 builder.add_node(...)
    - START → init_state 的 conditional_edges
    - init_state → classify_intent_or_skip (passthrough) 的 edge
    - classify_intent_or_skip → classify_intent | select_target_word 的 conditional_edges
    - classify_intent → evaluate | lookup_target | lookup_asked | skip 的 conditional_edges
    - update_mastery → format_output_or_branch (passthrough)
    - format_output_or_branch → select_target_word | explain_meaning | ... 的 conditional_edges
    - 各路径汇聚到 persist_turn → END
    """

async def build_agent(
    llm_flash: BaseChatModel,
    llm_smart: BaseChatModel,
    checkpointer: Any = None,
) -> CompiledStateGraph:
    """工厂：load_config() → 新建 KETPartnerAgent → StateGraph(BTPKetState)
    → wire_graph(builder, agent) → builder.compile(checkpointer=checkpointer)
    → graph.agent = agent (暴露给 shutdown 调 aclose) → 返回 graph。
    删除原 db 参数（函数体内从未使用）。
    """
```

### `sentence_orchestration.py` (NEW)

**职责**：句子生成与校验的编排逻辑，从 agent.py 抽出。无状态纯函数。

```python
async def generate_with_fallback(
    llm_smart: BaseChatModel,
    initial_target: str,
    initial_context: str,
    avoid_words: list[str],
    avoid_sentences: list[str],
    age: int,
    profile: dict,
    repos: KETPartnerRepos,
    config: KetConfig,
) -> tuple[str, ValidationResult, str, str]:
    """生成句子并在校验失败时重试；持续 naturalness 失败时切换 target word。

    循环：
      1. 至多 validate_retry_limit 次：
         a. generate_sentence(llm_smart, ...)
         b. validate_and_categorize(...)
         c. passed → 返回 (sentence, result, target, ctx)
         d. 否则记入 attempts，累积 seen_non_ket_words，重生成
      2. 最终仍未过：
         - 全是 non_ket_overflow → 选 non_ket_count 最少的 attempt 接受
         - 全是 naturalness 且未切过词 → 切换 target 后整轮重试
         - 否则接受当前草稿，warning 日志

    返回 (final_sentence, final_validation_result, final_target, final_context)。
    提取自 KETPartnerAgent._generate_with_fallback（110+ 行）。
    """

async def validate_and_categorize(
    llm_smart: BaseChatModel,
    sentence: str,
    target: str,
    age: int,
    repos: KETPartnerRepos,
    avoid_sentences: list[str],
) -> dict:
    """校验单条草稿，分类失败原因。

    流程：
      1. validate_sentence(sentence, repos, target) 拿 ValidationResult
      2. 计算 is_duplicate / non_ket_count / is_target_split
      3. 决定 passed 与 reason_kind：
         - non_ket_count <= 1 且不重复且未裂多词 target：
           - non_ket_count == 0 → 调 check_naturalness(llm_smart, sentence, age)
           - non_ket_count == 1 → 直接 passed
         - 否则按 is_target_split / is_duplicate / non_ket_overflow 分类

    返回 dict 含字段：
      result: ValidationResult
      passed: bool
      reason_kind: 'naturalness' | 'target_split' | 'duplicate' | 'non_ket_overflow' | None
      reason_detail: str
      non_ket_words: list[str]
      non_ket_count: int
      is_duplicate: bool
      is_target_split: bool
      sentence: str

    提取自 KETPartnerAgent._validate_and_categorize。
    """

def apply_multiword_target_patch(
    target: str,
    sentence: str,
    result: ValidationResult,
) -> None:
    """就地修改 result.words_used / result.non_ket_words：
    若 target 是多词（含空格）且出现在 sentence 中但未在 result.words_used，
    则追加 target 并剔除被 target 包含的子词。

    提取自 KETPartnerAgent.generate_sentence_node 内联块。
    """
```

### `mastery.py` (改名自 nodes.py)

**职责**：mastery 更新业务规则。

```python
async def apply_mastery_updates(state: BTPKetState, repos: KETPartnerRepos) -> None:
    """根据 intent + 评估结果更新 vocab_stats mastery。

    分支：
      - translation：
        - wrong_words 为空且 overall_correct is False → 全词 delta=0（neutral），早返回
        - 否则逐词 last_sentence_words：
          - 词在 wrong 集合 → delta=-1
          - 否则 delta=+1
          - target 词用 target_context，scaffolding 用 ''
      - idk：last_target_word 的 delta=-1
      - asks_meaning：asked_word 经 get_ket_word_any_context 规范化后 delta=-1
        （规范形式与 target 比较决定 context）
      - off_topic / non_compliant：no-op

    类型收紧：state 由 dict 改为 BTPKetState（CLAUDE.md §二.1）。
    """
```

### `output_format.py` (改名自 nodes.py)

**职责**：AI 回复文本渲染。

```python
def format_output_text(state: BTPKetState, new_sentence: str) -> str:
    """根据 intent + state 渲染回复文本。

    分支：
      - translation + wrong_words 非空：
        '正确翻译：X' / '你的翻译有误:' / 逐条 ' word 的意思是：correct'
      - translation + overall_correct is False 且无 wrong_words：
        '正确翻译：X' / '你的翻译和原句意思有些偏差。'
      - idk：'正确翻译：X'（若有）
        通用尾：'请把这句译成中文:' / '"new_sentence"' / 非 KET 词注释
      - non_ket_annotations 每条：'word 的意思是：meaning'

    类型收紧：state 由 dict 改为 BTPKetState。
    """
```

### `nodes.py` — **删除**

`apply_mastery_updates` 迁到 `mastery.py`，`format_output_text` 迁到 `output_format.py`，文件清空后删除。

### 现有文件不动

- `state.py`：`BTPKetState` TypedDict + `KetIntent` Literal
- `config.py`：`KetConfig` Pydantic 模型 + `load_config()`
- 9 个 LLM 领域模块：`input_classifier` / `sentence_generator` / `sentence_validator` / `sentence_naturalness` / `translation_evaluator` / `word_meaning_lookup` / `vocab_selector` / `multi_word_target` / `profile_summarizer`

### 迁出文件（从 `flow/ket_partner/` 删除）

- `db.py` → 拆到 `persistence/` 5 个文件
- `main.py` → `cli/ket_partner/main.py`
- `commands.py` → `cli/ket_partner/commands.py`
- `chat_logger.py` → `cli/ket_partner/chat_logger.py`
- `exporter.py` → `reporting/ket_partner/exporter.py`（同时拆出 `categories.py` + `markdown.py`）

---

## 五、`src/cli/ket_partner/` 详细规范

### `main.py`

**职责**：CLI 入口 —— init_db + 建 Repos + build_agent + 交互式聊天循环。

```python
async def main() -> None:
    """读取环境变量 (KID_NICKNAME / KID_AGE / KET_DB_PATH) →
    init_db() → Repos.for_user(db, 'default') → repos.log.append_session_start() →
    build_agent(llm_flash, llm_max) → ChatLogger.start_session() →
    CommandHandler(repos, chat_logger) →
    while True: asyncio.to_thread(input) →
      - 命令 (/开头) → CommandHandler.handle()，ExitLoop 则 break
      - 否则 messages.append(HumanMessage) → agent.ainvoke({messages}, config)
        → ai_reply = response['messages'][-1].content → print + chat_logger.log_turn
    finally: agent.agent.aclose() + db.close() + chat_logger.close_session()
    """

if __name__ == "__main__":
    asyncio.run(main())
```

**import 变更**：
- `from flow.ket_partner.agent import build_agent` → `from flow.ket_partner.graph import build_agent`
- `from flow.ket_partner.db import Repos, init_db` → `from persistence import Repos, init_db`
- `from flow.ket_partner.chat_logger import ChatLogger` → `from cli.ket_partner.chat_logger import ChatLogger`
- `from flow.ket_partner.commands import CommandHandler, ExitLoop` → `from cli.ket_partner.commands import CommandHandler, ExitLoop`
- `build_agent(llm_flash, llm_max, db)` → `build_agent(llm_flash, llm_max)`（db 参数删除）

### `commands.py`

**职责**：CLI 命令分发。

```python
class ExitLoop(Exception):
    """/exit /quit 抛出以打断主循环。"""


class CommandHandler:
    SUPPORTED: ClassVar[dict[str, str]] = {
        "/exportstats": "导出学习状态报告",
        "/exit":        "退出练习",
        "/quit":        "退出练习",
        "/help":        "显示命令列表",
    }

    def __init__(
        self,
        repos: Repos,                # ★ 改：原为 db_path: str
        chat_logger: ChatLogger,
    ) -> None:
        """复用主连接；不再单独 init_db。"""

    async def handle(self, user_input: str) -> None:
        """分发 /command：
          - /exit /quit → raise ExitLoop()
          - /help → _print_help()
          - /exportstats → await _export_stats()
          - 其他 → print '未知命令' 提示
        """

    def _print_help(self) -> None:
        """打印 SUPPORTED 表。"""

    async def _export_stats(self) -> None:
        """★ 改造：直接用 self.repos 调 reporting.ket_partner.exporter.export_learning_report。
        不再 init_db / 不再访问 repos._db 私有属性。
        """
```

### `chat_logger.py`

**职责**：CLI 会话日志写文件。**无逻辑改动**，纯文件迁移。

```python
class ChatLogger:
    def __init__(self, log_dir: str) -> None:
        """建 log_dir 目录；初始化 _fp=None / _session=None。"""

    def start_session(self, nickname: str) -> None:
        """按 chat_log_NNNN.txt 命名开新文件，写表头。"""

    def log_turn(self, turn_id: int, role: str, content: str) -> None:
        """写一行 '[turn NNNN - ROLE] content'。"""

    def close_session(self) -> None:
        """写尾部 + 关 fp。"""

    def _next_index(self) -> int:
        """扫 log_dir 找最大编号 + 1。私有。"""
```

---

## 六、`src/reporting/ket_partner/` 详细规范

### `categories.py` (NEW) — 分类规则唯一来源

**职责**：5 个 category 的判定规则，纯 Python 函数，无 DB 依赖。

```python
from typing import Literal
from flow.ket_partner.config import KetConfig

Category = Literal["mastered", "learning", "struggling", "used", "unused"]

CATEGORIES: tuple[str, ...] = ("mastered", "learning", "struggling", "used", "unused")


def classify_row(
    row: dict,
    struggling_wc_min: int,
    struggling_ec_min: int,
) -> Category:
    """对单条 stats row 分类。规则：
      - exposed_count == 0                                          → 'unused'
      - status == 'mastered'                                        → 'mastered'
      - status == 'learning'                                        → 'learning'
      - wrong_count >= struggling_wc_min
        OR (exposed_count >= struggling_ec_min AND mastery_score==0) → 'struggling'
      - else                                                        → 'used'
    """


def classify(row: dict, cfg: KetConfig) -> Category:
    """便捷包装：用 cfg.struggling_threshold.wrong_count_min / exposed_count_min。"""


def group_by_category(
    rows: list[dict],
    cfg: KetConfig,
) -> dict[str, list[dict]]:
    """一次性把 rows 分到 5 个 category 桶。
    返回 dict[category_name, rows]，键集 = CATEGORIES。
    """
```

**与 StatsRepo 的关系（Option B 已统一）**：原 `StatsRepo._category_where_sql` / `count_by_category` / `list_by_category` 已删除。StatsRepo 只保留 `list_all_with_vocab()`；分类规则全部走 `categories.py`。CLI 与 API `/report` 共享同一规则源。

### `markdown.py`

**职责**：Markdown 渲染辅助，纯函数。

```python
def fmt_word(word: str, context: str) -> str:
    """渲染：context 非空 → 'word(context)'；否则 → 'word'。"""


def render_markdown(
    profile: dict,
    rows_by_category: dict[str, list[dict]],
) -> str:
    """根据 profile + 已分桶 rows 渲染 Markdown 报告字符串。
    5 个 section：正在学习 / 已掌握 / 已使用 / 未使用 / 学习困难。
    接收预分桶数据（由 categories.group_by_category 产出），自身不做分类。
    """
```

### `exporter.py`

**职责**：报告导出编排。

```python
from flow.ket_partner.config import KetConfig
from flow.ket_partner.persistence import KETPartnerRepos
from reporting.ket_partner.categories import group_by_category
from reporting.ket_partner.markdown import render_markdown

async def export_learning_report(
    output_path: str,
    repos: KETPartnerRepos,        # ★ 类型由 Repos 改为 Protocol
    cfg: KetConfig,
    fmt: str = "markdown",
) -> str:
    """拉 stats rows → group_by_category → render_markdown → 写文件。
    返回 output_path。fmt 仅支持 'markdown'；其他抛 ValueError。
    """

async def render_report_text(
    repos: KETPartnerRepos,
    cfg: KetConfig,
) -> str:
    """★ NEW：内存版（不写文件），返回 Markdown 字符串。
    供未来 API /report 路由或单测直接拿渲染结果。
    """
```

**关键修复**：
| 原行为 | 新行为 |
|---|---|
| `async with repos.stats._db.execute(...)` 私有访问 | `await repos.stats.list_all_with_vocab()` 公开方法 |
| 内联 `_classify(row, cfg)` | `group_by_category(rows, cfg)` 来自 categories.py |
| `_render_markdown(profile, rows, cfg)` 内联分类+渲染 | `render_markdown(profile, rows_by_category)` 纯渲染 |
| 参数类型 `repos: Repos` | `repos: KETPartnerRepos` (Protocol) |

---

## 七、跨包契约与 Composition Root

### KETPartnerRepos Protocol

定义在 `flow/ket_partner/persistence.py`（见 §四）。`persistence/repos.py::Repos` 结构性满足该 Protocol；运行时无需显式 `Protocol` 注册。

### Composition Root

#### API（`src/api/app.py::lifespan` + `src/api/routes/chat.py`）

```python
# lifespan（启动一次）
db = await init_db(settings.DB_PATH, csv_path=csv_path, ...)
checkpointer = AsyncSqliteSaver(db); await checkpointer.setup()
agent = await build_agent(llm_flash, llm_max, checkpointer=checkpointer)
app.state.db = db; app.state.agent = agent

# routes/chat.py（每请求）
repos = Repos.for_user(db, user.id)
result = await agent.ainvoke(
    {"messages": [HumanMessage(content=req.text)]},
    config={"configurable": {
        "thread_id": f"{user.id}:main",
        "user_id": user.id,
        "repos": repos,                              # ← 注入点
        "user_info": {"nickname": user.nickname, "age": user.age},
    }},
)
```

#### CLI（`src/cli/ket_partner/main.py::main`）

```python
db = await init_db(db_path, csv_path=csv_path, ...)
repos = Repos.for_user(db, "default")
await repos.log.append_session_start()
agent = await build_agent(llm_flash, llm_max)        # ★ 不再传 db
# 后续 ainvoke 同样把 repos 放进 config["configurable"]["repos"]
```

#### API `/report` 路由（Option B 改造）

```python
# 原本：count_by_category + list_by_category 两次 SQL
# 改为：list_all_with_vocab() 一次 SQL + Python 过滤分页
from flow.ket_partner.config import load_config
from reporting.ket_partner.categories import CATEGORIES, group_by_category

cfg = load_config()
rows = await repos.stats.list_all_with_vocab()
bucket = group_by_category(rows, cfg)

# /report (总览)
return ReportResponse(
    mastered_count=len(bucket["mastered"]),
    learning_count=len(bucket["learning"]),
    struggling_count=len(bucket["struggling"]),
    used_count=len(bucket["used"]),
    unused_count=len(bucket["unused"]),
    total_words=await repos.vocab.total_count(),
)

# /report/{category} (分页)
if category not in CATEGORIES: raise HTTPException(400, ...)
category_rows = bucket[category]
total = len(category_rows)
total_pages = max(1, (total + page_size - 1) // page_size)
offset = (page - 1) * page_size
page_rows = category_rows[offset:offset + page_size]      # Python 切片分页
return ReportCategoryResponse(category=category, page=page, page_size=page_size,
                              total=total, total_pages=total_pages,
                              words=[ReportWord(**r) for r in page_rows])
```

### API 层源文件 import 更新清单

| 文件 | 原 import | 新 import |
|---|---|---|
| `src/api/app.py` | `from flow.ket_partner.agent import build_agent` | `from flow.ket_partner.graph import build_agent` |
| `src/api/app.py` | `from flow.ket_partner.db import init_db` | `from persistence import init_db` |
| `src/api/app.py` | `agent = await build_agent(llm_flash, llm_max, db, checkpointer=...)` | `agent = await build_agent(llm_flash, llm_max, checkpointer=...)` |
| `src/api/routes/chat.py` | `from flow.ket_partner.db import Repos` | `from persistence import Repos` |
| `src/api/routes/report.py` | `from flow.ket_partner.db import Repos` | `from persistence import Repos` |
| `src/api/routes/report.py` | (无) | `from flow.ket_partner.config import load_config` + `from reporting.ket_partner.categories import CATEGORIES, group_by_category`（Option B 改造） |

`/chat` 路由 body 不动；`/messages` / `/llm` 路由不动。

### 跨包测试边界

| 测试目录 | 被测包 | 不允许依赖 |
|---|---|---|
| `tests/persistence/` | persistence/ | flow/ket_partner/agent / cli / reporting |
| `tests/flow/ket_partner/` | flow/ket_partner/ | persistence 具体 impl（用 Mock 实现 Protocol） |
| `tests/cli/ket_partner/` | cli/ket_partner/ | — |
| `tests/reporting/ket_partner/` | reporting/ket_partner/ | cli / api |
| `tests/api/` | api/ | — |

---

## 八、测试重组策略

### 目标测试树

```
tests/
├── api/                                  # 现有
├── integration/                          # 现有
│   └── test_graph_integration.py         # ★ 移入：原 tests/ket_partner/ 下的 88K 怪兽
├── persistence/                          # ★ NEW
│   ├── conftest.py
│   ├── test_models.py                    # derive_status（从 test_db.py 拆）
│   ├── test_repos.py                     # 5 Repo + Repos（从 test_db.py 拆）
│   ├── test_bootstrap.py                 # init_db / _import_csv（从 test_db.py 拆）
│   └── test_migration.py                 # migrate_old_schema_if_needed（从 test_db.py 拆）
├── flow/ket_partner/                     # ★ 保留原子集
│   ├── conftest.py
│   ├── test_agent.py                     # KETPartnerAgent 节点单测（mock Protocol）
│   ├── test_graph.py                     # build_agent / wire_graph / 路由函数
│   ├── test_persistence_protocol.py      # ★ NEW: 验证 Repos 结构性满足 KETPartnerRepos
│   ├── test_sentence_orchestration.py    # ★ NEW: generate_with_fallback / validate_and_categorize
│   ├── test_mastery.py                   # 改名自 test_nodes.py（拆）
│   ├── test_output_format.py             # 改名自 test_nodes.py（拆）
│   ├── test_state.py / test_config.py    # 不动
│   └── (9 个 LLM 领域模块测试)            # import 路径不变
├── cli/ket_partner/                      # ★ NEW
│   ├── test_main.py                      # ★ NEW: main() async 测试
│   ├── test_commands.py                  # 移入 + 改 CommandHandler 构造签名
│   └── test_chat_logger.py               # 移入（无改动）
└── reporting/ket_partner/                # ★ NEW
    ├── test_exporter.py                  # 移入 + import 改 Protocol + list_all_with_vocab
    ├── test_categories.py                # ★ NEW: 纯 Python 单测 classify_row
    └── test_markdown.py                  # ★ NEW: 纯渲染测试
```

### 现有测试迁移清单

| 现文件 | 操作 | 关键改动 |
|---|---|---|
| `test_db.py` (28K) | **拆 4 个文件**到 `tests/persistence/` | `test_models.py` / `test_repos.py` / `test_bootstrap.py` / `test_migration.py`。**删除**：`_category_where_sql` / `count_by_category` / `list_by_category` 三类测试（Option B 删方法） |
| `test_graph_integration.py` (88K, 1900+ 行) | **移到 `tests/integration/`** + 大改 | 见下方 monkeypatch 迁移表 |
| `test_graph.py` | 留在 `tests/flow/ket_partner/` | `from flow.ket_partner.agent import build_agent` → `from flow.ket_partner.graph import build_agent` |
| `test_nodes.py` | **拆 2 个文件** | `test_mastery.py` + `test_output_format.py`；类型断言适配收紧后的 `BTPKetState` |
| `test_commands.py` | 移到 `tests/cli/ket_partner/` | `CommandHandler(db_path, ...)` 构造签名改 `CommandHandler(repos, ...)` |
| `test_chat_logger.py` | 移到 `tests/cli/ket_partner/` | 无改动 |
| `test_exporter.py` | 移到 `tests/reporting/ket_partner/` | `from flow.ket_partner.exporter import ...` → `from reporting.ket_partner.exporter import ...`；fixture 用 Protocol mock |
| `test_vocab_selector.py` / `test_sentence_validator.py` / `test_profile_summarizer.py` | 留在 `tests/flow/ket_partner/` | `from flow.ket_partner.db import Repos, init_db` → `from persistence import Repos, init_db` |
| `test_state.py` / `test_config.py` | 留在 `tests/flow/ket_partner/` | 无改动 |
| 9 个 LLM 领域模块测试 | 留在 `tests/flow/ket_partner/` | 无改动 |

### `test_graph_integration.py` monkeypatch 迁移映射

agent.py 拆分后，原本导入到 agent.py 命名空间的函数会散到 sentence_orchestration.py。

| 原 monkeypatch 目标 | 新目标 |
|---|---|
| `agent_module.generate_sentence` | `sentence_orchestration_module.generate_sentence` |
| `agent_module.validate_sentence` | `sentence_orchestration_module.validate_sentence` |
| `agent_module.check_naturalness` | `sentence_orchestration_module.check_naturalness` |
| `agent_module.target_in_sentence` | `sentence_orchestration_module.target_in_sentence` |
| `agent_module.evaluate_translation` | 不变（仍在 agent.py 命名空间） |
| `agent_module.classify_intent` | 不变 |
| `agent_module.lookup_word_meanings` | 不变 |
| `agent_module.lookup_sentence_translation` | 不变 |
| `agent_module.lookup_word_meaning` | 不变 |
| `agent_module.select_target_word` | 不变 |
| `agent_module.run_profile_summary` | 不变 |
| `agent_module.vocab_selector.select_target_word` | 不变（部分测试直接 patch 子模块） |

**死板规则**：每个 `agent_module.<name>` 出现处必须按映射表替换。**禁止**保留旧 import 形式。

### 新增的测试文件

| 文件 | 覆盖范围 |
|---|---|
| `tests/persistence/test_models.py` | `derive_status` 5 条边界：mastered / learning / exposed / 旧 mastered 降级 / is_target |
| `tests/flow/ket_partner/test_persistence_protocol.py` | 用 `runtime_checkable` 验证 `isinstance(Repos(...), KETPartnerRepos)`；检查 Protocol 方法集合是 Repos 公开方法的子集 |
| `tests/flow/ket_partner/test_sentence_orchestration.py` | `generate_with_fallback`：naturalness 失败切词 / non_ket_overflow 兜底 / 重复检测；`validate_and_categorize`：5 种 reason_kind |
| `tests/reporting/ket_partner/test_categories.py` | 5 个 category 各自的 row → category 分类（pure Python，无 DB） |
| `tests/reporting/ket_partner/test_markdown.py` | `render_markdown` 5 个 section 渲染（pure Python） |
| `tests/cli/ket_partner/test_main.py` | `main()` async 测试（mock build_agent + init_db，验证 ainvoke 调用 + cleanup） |

---

## 九、顺手清理

| # | 位置 | 操作 | 理由 |
|---|---|---|---|
| 1 | `src/flow/agent.py` | **删除整个文件** | `Autonomous` / `Consensus` 类全仓库无 import |
| 2 | `src/flow/log/` | **删除空目录** | 空目录，无任何 import |
| 3 | `src/flow/common.py` L16-19 | 删 `IS_RUNNING_IN_PYTEST` | 全仓库无引用 |
| 4 | `src/flow/common.py` L61-72 | 删 `llm_plus` 客户端 | 全仓库无 `import llm_plus` |
| 5 | `src/flow/common.py` L101 | 删 `doubao_api = environ.get(...)` | 仅喂给 llm_doubao |
| 6 | `src/flow/common.py` L102-116 | 删 `llm_doubao` 客户端 | 全仓库无 `import llm_doubao` |
| 7 | `src/flow/common.py` L98 | `extra_body={"enable_thinking": False}` → `extra_body=extra_params` | 与 llm_max 统一（CLAUDE.md §四.3） |
| 8 | `flow/ket_partner/agent.py::autonomous()` | **删除函数** | 全仓库无调用 |
| 9 | `flow/ket_partner/agent.py::build_agent(db 参数)` | **删除参数** | 函数体内从未使用（迁到 graph.py 时一并清理） |
| 10 | `flow/ket_partner/db.py::_derive_status` | 改名 `derive_status` | 跨模块调用，需公开 |
| 11 | `flow/ket_partner/db.py::_migrate_old_schema_if_needed` | 改名 `migrate_old_schema_if_needed` | 跨模块调用，需公开 |
| 12 | `flow/ket_partner/db.py::_import_csv` | 保持私有 `_import_csv` | 仅同模块（bootstrap.py）内调用 |

清理后的 `flow/common.py` 仅保留 4 个外部引用名字：`logger` / `llm_flash` / `llm_max` / `dashscope_api_key`，加内部常量 `extra_params`。

---

## 十、不在本次范围

### §10.1 现状声明（scope 边界，非问题）

给实现者的「别顺手改」清单。这些条目本身都不是问题，仅声明不在本次重构范围。

| 类别 | 不动项 |
|---|---|
| 工具 / 库 | pytest + pytest-asyncio / unittest.mock（不替换）；aiosqlite（不引入 SQLAlchemy 等）；LangGraph `AsyncSqliteSaver` + `MemorySaver` checkpointer；dashscope OpenAI-compatible 接口；`logging.getLogger("ket_partner")` |
| schema / 数据 | DB 表结构（不新增表/列、不重命名字段；仅 `migrate_old_schema_if_needed` 处理旧版兼容）；`ket_partner.db` 用户本地数据；`flow/ket_partner/data/*.json`（config / function_words / lemmas / seed_words） |
| 业务规则 | MASTERY_CAP / +1/-1 delta / `derive_status` 阈值；句子生成与校验算法；KetIntent 5 取值与分类 prompt；profile_summarizer 触发周期；KetConfig 字段与 config.json 内容 |
| API 路由 | 除 `/report` 外的路由不动（`/chat` / `/messages` / `/llm` 仅 import 更新）；`/report` 仅切分类源，响应 shape 不变；API 错误处理 / timeout / 认证 guard 不动 |
| Web 前端 | `web/` 完全不动 |
| LLM 领域模块内部 | 9 个领域模块（`sentence_generator` 等）**只搬不拆**，内部结构不动 |
| 测试基础设施 | 不替换 pytest；不引入新 mock 库；conftest fixture 体系仅镜像复制到新目录 |

### §10.2 已知技术债（本次不修的真问题，仅 1 条）

| # | 项 | 为什么是问题 | 为什么本次不修 |
|---|---|---|---|
| 1 | `passthrough_node` LangGraph hack（`flow/ket_partner/graph.py`） | `add_conditional_edges` 必须先有 node 才能挂分支，被迫塞 no-op 节点是框架设计限制 | 改掉要换框架或大改图拓扑，超本次范围 |

> 注：原本 §3 草稿中存在的「`persistence/repos.py` import `flow.ket_partner.config.KetConfig`」反向依赖，在 Option B 删除 `StatsRepo._category_where_sql` 后自动消除（StatsRepo / Repos 均不再需要 KetConfig 参数），故不再列入技术债。

---

## 十一、验收标准

- [ ] `ruff check src/ tests/` 全部清零
- [ ] `mypy src/ tests/` 全部清零
- [ ] `pytest tests/` 全部通过
- [ ] `flow/ket_partner/` 内无任何对**顶层 `persistence` / `cli` / `reporting` 包**的运行时 import（即禁止 `from persistence ...` / `from cli ...` / `from reporting ...`）。本地模块 `flow/ket_partner/persistence.py` 通过 `from flow.ket_partner.persistence import ...` 或 `from .persistence import ...` 引用，不在此约束内。`WordRef` 通过 `TYPE_CHECKING` 引用
- [ ] `persistence/` 内无任何 `from flow.ket_partner...` / `from cli` / `from reporting` import（Option B 删除分类 SQL 方法后，persistence 对 KetConfig 零依赖）
- [ ] `reporting/ket_partner/` 内无任何 `from cli` / `from api` import
- [ ] `flow/agent.py` 与 `flow/log/` 已删除
- [ ] `flow/common.py` 中 `IS_RUNNING_IN_PYTEST` / `llm_plus` / `llm_doubao` / `doubao_api` 已删除
- [ ] `exporter.py` 中无 `repos.stats._db.execute` 私有访问
- [ ] `commands.py::_export_stats` 中无 `init_db` / `repos._db.close()` 调用
- [ ] `StatsRepo` 公开方法集合不再包含 `_category_where_sql` / `count_by_category` / `list_by_category`
- [ ] `tests/integration/test_graph_integration.py` 所有 `agent_module.<name>` monkeypatch 已按迁移表替换
