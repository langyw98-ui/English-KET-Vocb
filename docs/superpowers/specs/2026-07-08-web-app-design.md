# KET Partner Web App Design

**Date:** 2026-07-08
**Status:** Design (awaiting plan)
**Owner:** 狂暴棕熊 + Claude

## Goal

把现有 CLI 形态的 KET Partner agent（`src/flow/ket_partner/`）包一层 FastAPI + Vue 3 Web 应用，让孩子能从浏览器使用，家长能在网页上看学习报告。

部署目标是 **A 阶段**（单用户、本地或家庭局域网），但 DB schema 与代码结构 **预埋多用户支持**，使得未来升级到 C 阶段（公开互联网、多用户注册）时业务代码无需重写，只需补账户 UI + 反向代理 + 限流。

## Non-goals（明确不做的事）

- **不做老 DB 兼容迁移**：现有 `ket_partner.db` 的开发期数据丢弃，重新初始化
- **不做流式响应**：`POST /api/chat` 返回完整 JSON，前端转圈等待 2-10s
- **不做 JWT / 真实登录**：A 阶段 `AUTH_MODE=disabled`，固定返回 default 用户
- **不做组件库**：前端手写 5-10 个轻量组件，不引 Element Plus / Naive UI
- **不做前端单元测试 / E2E**：A 阶段手动测够
- **不做 Docker / nginx**：单进程 uvicorn 直接服务前端静态文件

## Tech Stack

| 层 | 选型 |
|---|---|
| Backend | FastAPI + uvicorn + aiosqlite + LangGraph + LangChain（已有） |
| LLM | qwen via dashscope（已有） |
| Checkpointer | `MemorySaver` → `AsyncSqliteSaver`（与业务表共用同一 sqlite 文件） |
| Frontend | Vue 3 + Vite + TypeScript + Pinia + Vue Router 4 |
| HTTP | 原生 fetch（不引 axios） |
| Test | pytest（已有） + httpx.AsyncClient（API 测试） |

## File Structure

```
英语KET/英语/
├─ src/
│  ├─ flow/                         ← 现有 agent 库 (按本文档重构)
│  │  ├─ agent.py                   ← MemorySaver → AsyncSqliteSaver
│  │  ├─ common.py
│  │  └─ ket_partner/
│  │     ├─ agent.py                ← KETPartnerAgent 改无状态
│  │     ├─ db.py                   ← _SCHEMA 改目标形态
│  │     ├─ exporter.py             ← _classify 复用, 提供 SQL 条件
│  │     └─ ...
│  └─ api/                          ← 新增, 与 flow 平级
│     ├─ __init__.py
│     ├─ app.py                     ← FastAPI 实例 + startup/shutdown
│     ├─ deps.py                    ← get_current_user / get_db / get_agent
│     ├─ schemas.py                 ← pydantic DTOs
│     ├─ settings.py                ← BaseSettings 读 env
│     ├─ routes/
│     │  ├─ chat.py
│     │  ├─ report.py
│     │  └─ messages.py
│     └─ main.py                    ← uvicorn 入口
├─ web/                             ← Vue 项目 (根目录)
│  ├─ package.json
│  ├─ vite.config.ts                ← dev: /api → :8000
│  ├─ tsconfig.json
│  ├─ index.html
│  └─ src/
│     ├─ main.ts
│     ├─ App.vue
│     ├─ router.ts                  ← /chat (默认) + /report
│     ├─ api/
│     │  ├─ client.ts               ← fetch wrapper
│     │  └─ types.ts                ← 与后端 DTO 对齐
│     ├─ stores/
│     │  ├─ chat.ts
│     │  └─ report.ts
│     ├─ components/
│     │  └─ WordListModal.vue
│     └─ views/
│        ├─ ChatView.vue
│        └─ ReportView.vue
├─ data/                            ← KET_vocabulary.csv (已有)
└─ tests/
   ├─ ket_partner/                  ← 现有, 改造以匹配新 schema
   └─ api/                          ← 新增
      ├─ test_chat.py
      ├─ test_report.py
      └─ test_messages.py
```

## §1 架构与运行时拓扑

```
浏览器（Kid 设备）
  Vue 3 + Vite + TS + Pinia
  - 聊天页 /chat
  - 报告页 /report
        │
        │  HTTP JSON (同源, 无 CORS)
        ▼
FastAPI 进程（uvicorn 单 worker, localhost:8000）
  ├─ /api/chat        POST  接收 kid 输入 → agent.ainvoke → JSON
  ├─ /api/report      GET   5 个分类 count + total_words
  ├─ /api/report/{c}  GET   某分类词列表 (分页)
  ├─ /api/messages    GET   最近 15 条对话历史
  └─ /                GET   服务 Vue dist/ (生产模式)
  Dependencies:
  ├─ get_current_user   (AUTH_MODE=disabled 时返 'default')
  └─ get_db / get_agent (app.state 单例)
        │
        │  async aiosqlite (单连接, 进程级共享)
        ▼
SQLite (ket_partner.db)
  5 张改/新表 + LangGraph checkpointer 3 张表 (同文件)
        ▲
        │  LangGraph checkpointer
        │  AsyncSqliteSaver (替换 MemorySaver)
        │
  KETPartnerAgent (无状态单例)
```

**关键决策：**

1. **单进程 uvicorn**：A 阶段单 worker，asyncio 单线程事件循环天然支持多协程并发。C 升级时换 gunicorn 多 worker，代码零改动。

2. **同源部署**：开发用 Vite dev server (5173) 走 `vite.config.ts` 代理 `/api` 到 8000；生产用 FastAPI `StaticFiles` 服务 `web/dist/`。**避免 CORS 配置**。

3. **AsyncSqliteSaver 替换 MemorySaver**：现有 `flow/agent.py:49` 的 `MemorySaver()` 在后端重启时丢 LangGraph 状态。换成 `AsyncSqliteSaver`，与业务表共用同一 sqlite 文件，后端重启后 checkpointer 状态保留。应用启动时须调用 `await checkpointer.setup()` 创建底层 checkpoint 表。同时在 `init_state` 节点保持对状态中 `messages` 的滑动窗口截断（如保留最近 10 条），防止在持久化 checkpointer 下上下文随着对话无限膨胀。

4. **无状态 Agent 单例**：见下节"Agent 重构"。

5. **`thread_id = f"{user_id}:main"`**：每个请求传 `config={"configurable": {"thread_id": f"{user_id}:main", ...}}`，多用户隔离。

### Agent 重构：无状态单例

现有 `KETPartnerAgent` 在 `self` 上挂了 per-session 状态，直接做单例无法支持多用户。重构为无状态。

**改动表：**

| 状态 | 现状 | 改造后 |
|---|---|---|
| `self.llm_flash / llm_smart / config` | 进程级共享 | 保留 |
| `self._bg_tasks` | 进程级 asyncio 任务集合 | 保留（仅 shutdown 时 drain） |
| `self.info`（nickname, age） | per-user | **删除**，从 `config["configurable"]["user_info"]` 取 |
| `self._recent_sentences` | per-session dedup 缓冲 | **删除**，移到 `recent_sentences` 表 |
| `self._recent_scaffolding` | per-session dedup 缓冲 | **删除**，运行时从 `recent_sentences` 表 tokenize 推导（用与 `sentence_validator.py` 相同的 `[A-Za-z']+` regex） |
| `self.repos` | 单例 Repos | **删除**，per-request 通过 `config["configurable"]["repos"]` 透传 |

**新构造签名：**

```python
class KETPartnerAgent:
    def __init__(self, llm_flash, llm_smart, config):
        self.llm_flash = llm_flash
        self.llm_smart = llm_smart
        self.config = config
        self._bg_tasks = set()
```

**节点签名变更：**

```python
# 改造前
async def init_state(self, state: BTPKetState) -> dict:
    profile = await self.repos.profile.get()
    ...

# 改造后
async def init_state(self, state: BTPKetState, config: RunnableConfig) -> dict:
    repos = config["configurable"]["repos"]
    profile = await repos.profile.get()
    ...
```

约 10 个节点函数加 `config` 参数；节点内 `self.repos.xxx` 改为 `repos.xxx`；`self.info` 改为 `config["configurable"]["user_info"]`。

### 并发模型

- 单 worker uvicorn = 单事件循环 = 单 Python 线程
- 多协程在 await 点交错，**无 GIL 竞争、无 race condition**
- A 阶段单用户，根本不会触发并发（即使开多 tab，相同 thread_id 被 checkpointer 串行化）
- C 升级瓶颈全部在基础设施层（SQLite writer 锁、dashscope 限流），不在 Agent 代码

## §2 数据层

### Schema（目标形态，`_SCHEMA` in `db.py`）

**全局只读表（不动）：**

```sql
CREATE TABLE IF NOT EXISTS ket_vocabulary (
    word    TEXT NOT NULL,
    context TEXT NOT NULL DEFAULT '',
    pos     TEXT,
    is_seed INTEGER DEFAULT 0,
    PRIMARY KEY (word, context)
);
CREATE INDEX IF NOT EXISTS idx_vocab_seed ON ket_vocabulary(is_seed);

CREATE TABLE IF NOT EXISTS ket_vocab_topics (
    word    TEXT NOT NULL,
    context TEXT NOT NULL DEFAULT '',
    topic   TEXT NOT NULL,
    PRIMARY KEY (word, context, topic)
);
CREATE INDEX IF NOT EXISTS idx_vocab_topics_topic ON ket_vocab_topics(topic);
CREATE INDEX IF NOT EXISTS idx_vocab_topics_lookup ON ket_vocab_topics(word, context);
```

**业务表（改 + 新增）：**

```sql
-- vocab_stats: 加 user_id, PK 改 (user_id, word, context)
CREATE TABLE IF NOT EXISTS vocab_stats (
    word              TEXT NOT NULL,
    context           TEXT NOT NULL DEFAULT '',
    user_id           TEXT NOT NULL DEFAULT 'default',
    exposed_count     INTEGER DEFAULT 0,
    correct_count     INTEGER DEFAULT 0,
    wrong_count       INTEGER DEFAULT 0,
    mastery_score     INTEGER DEFAULT 0,
    status            TEXT DEFAULT 'new',
    first_seen_at     TIMESTAMP,
    last_seen_at      TIMESTAMP,
    PRIMARY KEY (user_id, word, context)
);
CREATE INDEX IF NOT EXISTS idx_stats_user_status ON vocab_stats(user_id, status);

-- conversation_log: 加 user_id
CREATE TABLE IF NOT EXISTS conversation_log (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id      TEXT NOT NULL DEFAULT 'default',
    role         TEXT,
    content      TEXT,
    words_used   TEXT,
    target_words TEXT,
    turn_id      INTEGER,
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_log_user_id ON conversation_log(user_id, id);

-- kid_profile: PK 改 user_id, 去掉 CHECK 约束 (学习状态表)
CREATE TABLE IF NOT EXISTS kid_profile (
    user_id            TEXT PRIMARY KEY,
    total_turns        INTEGER DEFAULT 0,
    weakness_words     TEXT,
    dialogue_strategy  TEXT,
    in_refill_mode     INTEGER DEFAULT 0,
    last_new_word_turn INTEGER DEFAULT 0,
    last_summary_turn  INTEGER DEFAULT 0,
    current_topic      TEXT,
    updated_at         TIMESTAMP
);

-- users: 新增 (身份表, C 升级时加 password_hash / email)
CREATE TABLE IF NOT EXISTS users (
    id          TEXT PRIMARY KEY,
    nickname    TEXT NOT NULL,
    age         INTEGER NOT NULL,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- recent_sentences: 新增 (替代 KETPartnerAgent._recent_sentences)
CREATE TABLE IF NOT EXISTS recent_sentences (
    user_id     TEXT NOT NULL,
    sentence    TEXT NOT NULL,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_recent_user_created ON recent_sentences(user_id, created_at DESC);
```

**注意：** `nickname` / `age` 移到 `users` 表（原 `kid_profile` 的这两列删除），保持身份信息与学习状态分离。C 升级时 `users` 成为账号锚点。

### session_start 机制 drop

现有 `LogRepo.append_session_start()` / `last_ai_message()` 用 `role='system', content='session_start'` 标记会话边界。Web 模式下**没有 session 概念**（kid 永远是从上一次继续），整个机制删除：

- `LogRepo.append_session_start` 删除
- `LogRepo.last_ai_message` 简化为 `SELECT ... WHERE role='ai' AND user_id=? ORDER BY id DESC LIMIT 1`
- `flow/ket_partner/main.py`（CLI 入口）里 `await repos.log.append_session_start()` 调用删除

### init_db 签名与 seed

```python
async def init_db(
    db_path: str,
    csv_path: Optional[str] = None,
    default_nickname: str = "宝贝",
    default_age: int = 8,
) -> aiosqlite.Connection:
    db = await aiosqlite.connect(db_path)
    db.row_factory = sqlite3.Row
    # 开启 WAL 模式与 5s 忙等待超时, 解决 async 并发读写 database is locked 问题
    await db.execute("PRAGMA journal_mode=WAL;")
    await db.execute("PRAGMA busy_timeout=5000;")
    await db.executescript(_SCHEMA)

    # 幂等 seed: 进程每次启动都跑; 首次插入, 后续 OR IGNORE 跳过
    await db.execute(
        "INSERT OR IGNORE INTO users (id, nickname, age) VALUES ('default', ?, ?)",
        (default_nickname, default_age),
    )
    await db.execute(
        "INSERT OR IGNORE INTO kid_profile (user_id, total_turns) VALUES ('default', 0)",
    )
    await db.commit()

    if csv_path:
        await _import_csv(db, csv_path)

    return db   # 返回 connection, 不再包 Repos
```

### Repo 层：per-request Repos

```python
class Repos:
    def __init__(self, db: aiosqlite.Connection, user_id: str):
        self._db = db
        self._user_id = user_id
        self.vocab = VocabRepo(db)                  # 只读, 不需 user_id
        self.stats = StatsRepo(db, user_id)
        self.profile = ProfileRepo(db, user_id)
        self.log = LogRepo(db, user_id)
        self.recent = RecentSentencesRepo(db, user_id)

    @classmethod
    def for_user(cls, db, user_id): return cls(db, user_id)
```

**所有 Repo 方法签名不再带 `user_id` 参数**（构造时绑定），SQL 内部用 `self._user_id`。

### RecentSentencesRepo（新增）

替代 `KETPartnerAgent._recent_sentences` 与 `_recent_scaffolding`：

```python
class RecentSentencesRepo:
    def __init__(self, db, user_id):
        self._db = db
        self._user_id = user_id

    async def list_recent(self, limit: int = 20) -> List[str]:
        """最近 N 句, 按 created_at DESC"""
        ...

    async def append(self, sentence: str, window: int = 20) -> None:
        """INSERT 新句 + 同事务 DELETE 超出窗口的旧行"""
        ...

    async def list_recent_scaffolding(self, window: int = 20) -> List[List[str]]:
        """从 recent_sentences tokenize 出 scaffolding 词列表
        用与 sentence_validator.py 相同的 [A-Za-z']+ regex"""
        ...
```

### StatsRepo 新增方法（支持报告分类）

```python
class StatsRepo:
    # ... 现有方法保留 ...

    async def count_by_category(self, category: str) -> int:
        """按分类返回词数"""
        sql, params = self._category_where_sql(category)
        async with self._db.execute(f"SELECT COUNT(*) FROM ({sql})", params) as cur:
            return (await cur.fetchone())[0]

    async def list_by_category(
        self, category: str, offset: int = 0, limit: int = 100
    ) -> List[dict]:
        """按分类返回词汇列表, 分页"""
        sql, params = self._category_where_sql(category)
        sql += " LIMIT ? OFFSET ?"
        params = (*params, limit, offset)
        ...

    def _category_where_sql(self, category: str) -> tuple[str, tuple]:
        """复用 exporter._classify 的判定逻辑转 SQL.

        struggling_threshold 从 KetConfig 读 (当前 data/config.json 的值是
        wrong_count_min=2, exposed_count_min=4), 通过参数传入 SQL, 不写死.
        """
        wc_min = self._config.struggling_threshold.wrong_count_min
        ec_min = self._config.struggling_threshold.exposed_count_min

        if category == "mastered":
            return "SELECT * FROM vocab_stats WHERE user_id=? AND status='mastered'", (self._user_id,)
        if category == "learning":
            return "SELECT * FROM vocab_stats WHERE user_id=? AND status='learning'", (self._user_id,)
        if category == "struggling":
            return (
                "SELECT * FROM vocab_stats WHERE user_id=? "
                "AND status NOT IN ('mastered', 'learning') "
                "AND exposed_count > 0 "
                "AND (wrong_count >= ? OR (exposed_count >= ? AND mastery_score = 0))",
                (self._user_id, wc_min, ec_min)
            )
        if category == "used":
            return (
                "SELECT * FROM vocab_stats WHERE user_id=? "
                "AND exposed_count > 0 "
                "AND status NOT IN ('mastered', 'learning') "
                "AND NOT (wrong_count >= ? OR (exposed_count >= ? AND mastery_score = 0))",
                (self._user_id, wc_min, ec_min)
            )
        if category == "unused":
            # 在词表但无 stats 行, 或 stats 行 exposed_count=0 (与 exporter._classify 一致)
            return (
                "SELECT v.word, v.context, COALESCE(s.mastery_score, 0) AS mastery_score, "
                "COALESCE(s.exposed_count, 0) AS exposed_count, "
                "COALESCE(s.correct_count, 0) AS correct_count, "
                "COALESCE(s.wrong_count, 0) AS wrong_count, "
                "COALESCE(s.status, 'new') AS status "
                "FROM ket_vocabulary v "
                "LEFT JOIN vocab_stats s ON s.word = v.word AND s.context = v.context AND s.user_id = ? "
                "WHERE s.word IS NULL OR s.exposed_count = 0",
                (self._user_id,)
            )
        raise ValueError(f"invalid category: {category}")
```

`StatsRepo` 构造需接收 `KetConfig`：`StatsRepo(db, user_id, config)`，与 `Repos` 持有 config 一致。

## §3 API 层

### Endpoints（4 个）

```
POST /api/chat                发送 kid 输入, 返回 AI 回复
GET  /api/report              返回 5 个 count + total_words
GET  /api/report/{category}   某分类词列表 (分页)
GET  /api/messages            最近 15 条对话历史
```

### 配置（`src/api/settings.py`）

```python
from pydantic_settings import BaseSettings
from typing import Literal, Optional

class Settings(BaseSettings):
    DB_PATH: str = "ket_partner.db"
    CSV_PATH: Optional[str] = None
    AUTH_MODE: Literal["disabled", "jwt"] = "disabled"
    KID_NICKNAME: str = "宝贝"
    KID_AGE: int = 8
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    REQUEST_TIMEOUT: int = 30

    class Config:
        env_file = ".env"
```

### DTOs（`src/api/schemas.py`）

```python
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

# --- Chat ---
class ChatRequest(BaseModel):
    text: str

class ChatResponse(BaseModel):
    ai_reply: str       # 与 CLI 输出一致的预格式化文本
    turn_id: int

# --- Report ---
class ReportResponse(BaseModel):
    mastered_count: int
    learning_count: int
    struggling_count: int
    used_count: int
    unused_count: int
    total_words: int    # 词表总数, 用于算百分比

class ReportWord(BaseModel):
    word: str
    context: str = ""
    mastery_score: int
    exposed_count: int
    correct_count: int
    wrong_count: int
    status: str

class ReportCategoryResponse(BaseModel):
    category: str
    page: int
    page_size: int
    total: int
    total_pages: int
    words: List[ReportWord]

# --- Messages ---
class MessageOut(BaseModel):
    role: str           # "user" | "ai" | "system"
    content: str
    turn_id: Optional[int] = None
    created_at: datetime
```

### JSON 契约示例（Payload Examples）

**1. POST /api/chat**
- 请求 (Request):
  ```json
  {
    "text": "猫在冰上飞"
  }
  ```
- 响应 (Response 200 OK):
  ```json
  {
    "ai_reply": "错了 1 个词：\n- slipped: 飞→滑倒\n正确翻译：猫在冰上滑倒了",
    "turn_id": 1
  }
  ```

**2. GET /api/report**
- 响应 (Response 200 OK):
  ```json
  {
    "mastered_count": 12,
    "learning_count": 8,
    "struggling_count": 3,
    "used_count": 45,
    "unused_count": 1432,
    "total_words": 1500
  }
  ```

**3. GET /api/report/learning?page=1&page_size=100**
- 响应 (Response 200 OK):
  ```json
  {
    "category": "learning",
    "page": 1,
    "page_size": 100,
    "total": 8,
    "total_pages": 1,
    "words": [
      {
        "word": "slip",
        "context": "slipping on ice",
        "mastery_score": 1,
        "exposed_count": 2,
        "correct_count": 1,
        "wrong_count": 1,
        "status": "learning"
      }
    ]
  }
  ```

**4. GET /api/messages?limit=15**
- 响应 (Response 200 OK):
  ```json
  [
    {
      "role": "user",
      "content": "猫在冰上飞",
      "turn_id": 1,
      "created_at": "2026-07-27T15:00:00.000Z"
    },
    {
      "role": "ai",
      "content": "错了 1 个词：\n- slipped: 飞→滑倒\n正确翻译：猫在冰上滑倒了",
      "turn_id": 1,
      "created_at": "2026-07-27T15:00:02.000Z"
    }
  ]
  ```

### `POST /api/chat` 实现

```python
@router.post("", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    user: User = Depends(get_current_user),
    db: aiosqlite.Connection = Depends(get_db),
    agent: CompiledStateGraph = Depends(get_agent),
    settings: Settings = Depends(get_settings),
):
    repos = Repos.for_user(db, user.id)
    user_info = {"nickname": user.nickname, "age": user.age}

    try:
        result_state = await asyncio.wait_for(
            agent.ainvoke(
                {"messages": [HumanMessage(content=req.text)]},
                config={
                    "configurable": {
                        "thread_id": f"{user.id}:main",
                        "user_id": user.id,
                        "repos": repos,
                        "user_info": user_info,
                    }
                },
            ),
            timeout=settings.REQUEST_TIMEOUT,
        )
    except asyncio.TimeoutError:
        raise HTTPException(504, "agent timeout")

    ai_text = result_state["messages"][-1].content

    # 注意: agent 内部 persist_turn_node 已把 user/ai 消息写入 conversation_log 并且更新了 total_turns;
    # 这里不能重复写 log.append, 直接从 profile 读取最新的 turn_id
    profile = await repos.profile.get()
    turn_id = profile.get("total_turns", 0)

    return ChatResponse(ai_reply=ai_text, turn_id=turn_id)
```

### `GET /api/report`

```python
@router.get("", response_model=ReportResponse)
async def report_counts(
    user: User = Depends(get_current_user),
    db: aiosqlite.Connection = Depends(get_db),
):
    repos = Repos.for_user(db, user.id)
    return ReportResponse(
        mastered_count=await repos.stats.count_by_category("mastered"),
        learning_count=await repos.stats.count_by_category("learning"),
        struggling_count=await repos.stats.count_by_category("struggling"),
        used_count=await repos.stats.count_by_category("used"),
        unused_count=await repos.stats.count_by_category("unused"),
        total_words=await repos.vocab.total_count(),
    )
```

### `GET /api/report/{category}`

```python
@router.get("/{category}", response_model=ReportCategoryResponse)
async def report_by_category(
    category: str,
    page: int = 1,
    page_size: int = 100,
    user: User = Depends(get_current_user),
    db: aiosqlite.Connection = Depends(get_db),
):
    valid = {"mastered", "learning", "struggling", "used", "unused"}
    if category not in valid:
        raise HTTPException(400, f"invalid category: {category}")
    if page < 1 or page_size < 1 or page_size > 500:
        raise HTTPException(400, "invalid pagination params")

    repos = Repos.for_user(db, user.id)
    total = await repos.stats.count_by_category(category)
    total_pages = max(1, (total + page_size - 1) // page_size)
    offset = (page - 1) * page_size
    rows = await repos.stats.list_by_category(category, offset=offset, limit=page_size)
    return ReportCategoryResponse(
        category=category,
        page=page,
        page_size=page_size,
        total=total,
        total_pages=total_pages,
        words=[ReportWord(**r) for r in rows],
    )
```

### `GET /api/messages`

```python
@router.get("", response_model=List[MessageOut])
async def messages(
    limit: int = 15,
    user: User = Depends(get_current_user),
    db: aiosqlite.Connection = Depends(get_db),
):
    if limit < 1 or limit > 100:
        raise HTTPException(400, "limit must be in [1, 100]")
    repos = Repos.for_user(db, user.id)
    rows = await repos.log.recent(limit=limit)
    return [MessageOut(**r) for r in rows]
```

### 依赖（`src/api/deps.py`）

```python
class User(BaseModel):
    id: str
    nickname: str
    age: int

async def get_current_user(request: Request) -> User:
    settings: Settings = request.app.state.settings
    if settings.AUTH_MODE == "disabled":
        # A 阶段 stub: 从 users 表读 'default' 行
        db = request.app.state.db
        async with db.execute(
            "SELECT id, nickname, age FROM users WHERE id='default'"
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            raise HTTPException(500, "default user not seeded")
        return User(id=row[0], nickname=row[1], age=row[2])
    # C 阶段: 解析 JWT, 查 users 表
    raise NotImplementedError("JWT auth not implemented yet")

async def get_db(request: Request) -> aiosqlite.Connection:
    return request.app.state.db

async def get_agent(request: Request) -> CompiledStateGraph:
    return request.app.state.agent

async def get_settings(request: Request) -> Settings:
    return request.app.state.settings
```

from contextlib import asynccontextmanager
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings()
    db = await init_db(
        settings.DB_PATH,
        csv_path=settings.CSV_PATH,
        default_nickname=settings.KID_NICKNAME,
        default_age=settings.KID_AGE,
    )
    # 初始化 AsyncSqliteSaver 并显式执行 setup 创建 checkpoints 依赖表
    checkpointer = AsyncSqliteSaver(db)
    await checkpointer.setup()

    agent = await build_agent(llm_flash, llm_max, db, checkpointer=checkpointer)
    app.state.settings = settings
    app.state.db = db
    app.state.agent = agent

    yield

    inner = getattr(agent, "agent", None)
    if inner is not None:
        await inner.aclose()
    await app.state.db.close()

app = FastAPI(lifespan=lifespan)

# 全局 exception handler
@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    logger.exception(f"unhandled error on {request.url}: {exc}")
    return JSONResponse(status_code=500, content={"detail": "internal error"})

# 路由注册
app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
app.include_router(report.router, prefix="/api/report", tags=["report"])
app.include_router(messages.router, prefix="/api/messages", tags=["messages"])

# 生产模式: 服务 Vue 构建产物
if Path("web/dist").exists():
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse

    @app.get("/")
    async def index():
        return FileResponse("web/dist/index.html")

    app.mount("/", StaticFiles(directory="web/dist", html=True), name="static")
```

## §4 前端

### 两个 view + 顶部 tab

```
┌────────────────────────────────────────┐
│  KET 英语搭子     [对话] [报告]        │  ← 顶部 tab (Vue Router)
├────────────────────────────────────────┤
│  <RouterView/>                         │
└────────────────────────────────────────┘
```

### ChatView

```
┌────────────────────────────────────────┐
│  AI: 请把这句译成中文:                  │
│  'The cat slipped on the ice.'         │  ← 消息列表 (滚动区)
│                                        │
│  我: 猫在冰上飞                         │
│                                        │
│  AI: 错了 1 个词：                      │
│      - slipped: 飞→滑倒                │
│      正确翻译：猫在冰上滑倒了           │
│                                        │
├────────────────────────────────────────┤
│  [输入框........................] [发送]│  ← 底部输入栏
└────────────────────────────────────────┘
```

行为：
- 进入页面 → `GET /api/messages?limit=15` 拉历史 → 渲染
- 输入 + 回车 / 点发送 → 乐观追加 user 消息 → `POST /api/chat` → 等待 2-10s（输入框禁用 + 转圈）→ 追加 AI 回复
- 发送失败：撤回乐观追加，输入框上方红字提示 + 重试按钮
- 自动滚到底部

### ReportView

```
┌────────────────────────────────────────┐
│  ┌────────┐ ┌────────┐ ┌────────┐     │
│  │ 已掌握  │ │ 正在学  │ │ 有困难 │     │  ← 5 张可点击 count 卡片
│  │   12   │ │    8   │ │    3   │     │
│  │ 0.8%   │ │ 0.5%   │ │ 0.2%   │     │
│  └────────┘ └────────┘ └────────┘     │
│  ┌────────┐ ┌────────┐                 │
│  │ 接触过  │ │ 未学   │                 │
│  │   45   │ │  1432  │                 │
│  └────────┘ └────────┘                 │
│                                        │
│           词表总数：1500               │
└────────────────────────────────────────┘
```

行为：
- 进入页面 → `GET /api/report` → 渲染 5 张卡片
- 点击卡片 → 打开 WordListModal → `GET /api/report/{category}?page=1&page_size=100`
- 不自动刷新

### WordListModal

```
┌──────────────────────────────────────────┐
│  正在学习 (8 词)                    [×]   │
├──────────────────────────────────────────┤
│  ┌────────────────────────────────────┐  │
│  │ cat                                │  │
│  │ 掌握度 1/2    错 1 次  对 2 次     │  │
│  └────────────────────────────────────┘  │
│  ...                                      │
├──────────────────────────────────────────┤
│  [< 上一页]   第 1 / 3 页   [下一页 >]   │  ← 分页栏 (单页时不显示)
└──────────────────────────────────────────┘
```

每词卡片显示 word + stats（未学分类下只显示 word）。`<Teleport to="body">` 渲染。点 × 或遮罩关闭。

### fetch wrapper（`web/src/api/client.ts`）

```typescript
// 开发模式下走 vite.config.ts 的 /api 代理, 生产模式下由 FastAPI 托管静态资源, 统一使用相对路径路径避免 CORS 跨域问题
const BASE = ''

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  })
  if (!res.ok) {
    const text = await res.text().catch(() => '')
    let detail = text
    try {
      const json = JSON.parse(text)
      detail = json.detail || text
    } catch {}
    throw new Error(`API ${res.status}: ${detail}`)
  }
  return res.json()
}
```

### 类型对齐（`web/src/api/types.ts`）

```typescript
export interface ChatResponse {
  ai_reply: string
  turn_id: number
}

export interface Message {
  role: 'user' | 'ai' | 'system'
  content: string
  turn_id: number | null
  created_at: string
}

export interface ReportResponse {
  mastered_count: number
  learning_count: number
  struggling_count: number
  used_count: number
  unused_count: number
  total_words: number
}

export interface ReportWord {
  word: string
  context: string
  mastery_score: number
  exposed_count: number
  correct_count: number
  wrong_count: number
  status: string
}

export interface ReportCategoryResponse {
  category: string
  page: number
  page_size: number
  total: number
  total_pages: number
  words: ReportWord[]
}
```

类型手写对齐后端 pydantic（不引 openapi-codegen）。

### Pinia stores

**chat store：**

```typescript
export const useChatStore = defineStore('chat', () => {
  const messages = ref<Message[]>([])
  const sending = ref(false)
  const error = ref<string | null>(null)

  async function load() {
    messages.value = await api<Message[]>('/api/messages?limit=15')
  }
  async function send(text: string) {
    sending.value = true
    error.value = null
    const optimistic: Message = {
      role: 'user', content: text, turn_id: null,
      created_at: new Date().toISOString()
    }
    messages.value.push(optimistic)
    try {
      const res = await api<ChatResponse>('/api/chat', {
        method: 'POST', body: JSON.stringify({ text })
      })
      optimistic.turn_id = res.turn_id
      messages.value.push({
        role: 'ai', content: res.ai_reply, turn_id: res.turn_id,
        created_at: new Date().toISOString()
      })
    } catch (e: any) {
      error.value = e.message
      messages.value.pop()  // 撤回乐观追加
    } finally {
      sending.value = false
    }
  }

  return { messages, sending, error, load, send }
})
```

**report store：**

```typescript
export const useReportStore = defineStore('report', () => {
  const counts = ref<ReportResponse | null>(null)
  const activeCategory = ref<string | null>(null)
  const words = ref<ReportWord[]>([])
  const page = ref(1)
  const pageSize = ref(100)
  const total = ref(0)
  const totalPages = ref(1)
  const loadingPage = ref(false)

  async function loadCounts() {
    counts.value = await api<ReportResponse>('/api/report')
  }
  async function openCategory(cat: string) {
    activeCategory.value = cat
    await loadPage(1)
  }
  async function loadPage(p: number) {
    if (!activeCategory.value) return
    loadingPage.value = true
    try {
      const res = await api<ReportCategoryResponse>(
        `/api/report/${activeCategory.value}?page=${p}&page_size=${pageSize.value}`
      )
      words.value = res.words
      page.value = res.page
      total.value = res.total
      totalPages.value = res.total_pages
    } finally {
      loadingPage.value = false
    }
  }
  function closeCategory() {
    activeCategory.value = null
    words.value = []
    page.value = 1; total.value = 0; totalPages.value = 1
  }
  async function nextPage() {
    if (page.value < totalPages.value) await loadPage(page.value + 1)
  }
  async function prevPage() {
    if (page.value > 1) await loadPage(page.value - 1)
  }

  return {
    counts, activeCategory, words, page, pageSize, total, totalPages, loadingPage,
    loadCounts, openCategory, loadPage, closeCategory, nextPage, prevPage,
  }
})
```

### `vite.config.ts`

```typescript
import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  server: {
    proxy: { '/api': 'http://localhost:8000' }
  }
})
```

开发时访问 http://localhost:5173，所有 `/api/*` 自动转发到 8000。

## §5 错误处理 / 测试 / 部署

### 错误处理

| 场景 | 处理 | HTTP |
|---|---|---|
| 请求体校验失败 | pydantic 自动 | 422 |
| `category` 不在白名单 | HTTPException(400) | 400 |
| 分页参数越界 | HTTPException(400) | 400 |
| LLM 调用失败 | agent 内部 try/except 兜底（已有） | 200 |
| Agent 整体超时（>30s） | `asyncio.wait_for` 包住，超时 HTTPException(504) | 504 |
| DB 错误 | 全局 exception handler 记录 + 500 | 500 |
| 未捕获异常 | 全局 exception handler 记 traceback + 500 | 500 |

**前端错误展示：**
- 聊天页发送失败：输入框上方红字提示 + 重试按钮
- 报告页加载失败：卡片区域"加载失败，点击重试"
- Modal 加载失败：modal 内"加载失败，点击重试"

### 测试策略

| 层 | 工具 | 范围 |
|---|---|---|
| Repo 单元测试（已有，改造） | pytest + aiosqlite 临时文件 | `Repos(db, "default")` 构造，验证 user_id 隔离（双用户场景） |
| API endpoint 测试（新） | pytest + httpx.AsyncClient | chat / report / report/{cat} / messages 的正常路径 + 4xx + 分页边界 |
| Agent 集成测试（已有，改造） | pytest | 节点签名加 config，从 config 取 repos |
| 前端 / E2E | 不做 | A 阶段 YAGNI |

**关键测试场景：**

- POST /api/chat：发送文本 → 返回 ai_reply + turn_id 自增
- GET /api/report：5 个 count 与 DB 实际分类一致
- GET /api/report/{category}：page=0 / page_size=501 → 400；page=999 → 空 words
- GET /api/messages?limit=15：返回最近 15 条按时间正序
- Repo 双用户：user_a 的 vocab_stats 不被 user_b 的查询看到

### 部署模式

**开发模式（双进程）：**

```bash
# 终端 1: 后端
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m uvicorn src.api.main:app --reload --port 8000

# 终端 2: 前端
cd web && npm run dev    # → http://localhost:5173
```

**生产模式（单进程）：**

```bash
cd web && npm run build              # 输出 web/dist/
D:/ProgramData/miniforge3/envs/langgraph/python.exe -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
# 访问 http://localhost:8000 或 http://<本机内网IP>:8000
```

无 HTTPS（A 阶段家庭局域网够用）。无 Docker / nginx。Windows 上若想常驻，用 nssm 或任务计划程序。

### Python 环境

所有 Python / pytest 命令用 venv：`D:/ProgramData/miniforge3/envs/langgraph/python.exe`。

### 日志

沿用 `flow.common.logger`（loguru），后端 API 请求/错误自动落 `src/flow/log/log.txt`。不加 Prometheus / OpenTelemetry。

## A→C 升级清单（不在本期做，但本期做的预埋工作使其成为增量）

| 升级项 | 本期是否预埋 |
|---|---|
| DB 表加 user_id 列 + 复合 PK | ✓ 已做 |
| API endpoint 通过 `get_current_user` 注入 user_id | ✓ 已做（stub） |
| `thread_id` 含 user_id | ✓ 已做 |
| `users` 表作为账号锚点（加 password_hash / email） | ✓ 表已存在，C 阶段加列 |
| `AUTH_MODE=disabled → jwt` 切换 | ✓ 已做（同一份代码） |
| `KETPartnerAgent` 无状态化 | ✓ 已做 |
| SQLite → Postgres（业务表 + checkpointer） | ✗ C 阶段换 |
| 单 worker → gunicorn 多 worker | ✗ C 阶段换 |
| nginx 反向代理 + TLS | ✗ C 阶段加 |
| dashscope 限流 | ✗ C 阶段加 |
| 注册 / 登录 UI | ✗ C 阶段加 |
| 数据导出 / 删除接口（合规） | ✗ C 阶段加 |

C 阶段没有任何"重写 Agent / Repo"工作，全部是基础设施层与 UI 层的增量。

## 已知约束 / 项目规范

- **编码规范准则**：严格遵循 [.agents/rules/code-standards.md](file:///D:/Workspace/HBuilderProjects/%E8%8B%B1%E8%AF%ADKET/%E8%8B%B1%E8%AF%AD/.agents/rules/code-standards.md) 中的全部开发纪律：
  1. 异常捕获禁止裸 `except` / `except Exception`；所有跨边界调用（LLM/HTTP/DB/文件）必须捕获具体异常；所有 fallback 必须带有 `logger.warning(..., exc_info=True)`；
  2. LLM 调用统一使用 `with_structured_output` + Pydantic Schema（且必须传 `method="function_calling"`）；
  3. 异步函数内禁止任何同步阻塞 IO，数据契约（Pydantic DTO）与路由逻辑分文件隔离；
  4. 单元测试必须具备 Hermetic 性，断言 mock 时必须校验 `call_count` / `assert_awaited_once`。
- **RTK 命令行前缀**：按照 [.agents/rules/antigravity-rtk-rules.md](file:///D:/Workspace/HBuilderProjects/%E8%8B%B1%E8%AF%ADKET/%E8%8B%B1%E8%AF%AD/.agents/rules/antigravity-rtk-rules.md)，所有终端命令统一使用 `rtk` 前缀（如 `rtk pytest`）。
- **Python venv**：`D:/ProgramData/miniforge3/envs/langgraph/python.exe`（base miniforge3 缺依赖）。
- **静态检查三项**：验证阶段必须同时通过 `ruff check`、`mypy`、`pytest`。
- **无 emoji**：用户终端 GBK 编码，emoji 渲染为乱码。日志、提示文案仅用纯中文 + 常见 CJK 标点，不要在 console / log 输出 emoji。
