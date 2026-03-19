# Nervon — Executive Summary
### Reasoning-Native Memory Framework for AI Agents
> Last updated: 2026-03-19 | Author: Russ (unicornnoway)

---

## 一句话

**Nervon 是一个让 AI agent 像人一样记忆的开源 SDK** — 不是把所有东西塞进向量数据库然后祈祷 cosine similarity 能找到对的东西，而是让 LLM 主动决定：记什么、忘什么、怎么更新、怎么解决矛盾。

---

## 为什么这个东西重要

AI agent 的记忆是个 **$150M 的问题**。Mem0 刚拿了 $24M 融资（估值 $150M），就靠一个向量数据库 + LLM 提取的简单架构。但所有现有方案都有同一个致命缺陷：

> **向量搜索找的是"相似的文字"，不是"相关的知识"。**

"用户住在纽约" 和 "用户从纽约搬到旧金山" 在向量空间里几乎一样 — 但一个已经过时了。没有任何现有系统能优雅地处理这个。

Nervon 的答案：**把 LLM 推理嵌入记忆的每个阶段** — 提取、对比、更新、检索。向量搜索降级为粗筛，LLM 做最终判断。

---

## 核心架构：三层记忆

不是一个花哨的比喻，是三种完全不同的访问模式决定的：

```
┌─────────────────────────────────────────────────┐
│  Working Memory（工作记忆）                       │
│  → 永远加载，零搜索成本                            │
│  → 用户名、偏好、当前任务（最多10条）                │
│  → 类比：你脑子里时刻记着的东西                      │
├─────────────────────────────────────────────────┤
│  Semantic Store（语义存储）                        │
│  → 按意义搜索                                     │
│  → 所有从对话中提取的事实                           │
│  → 带时间戳版本控制（旧的不删，标注"已过期"）          │
│  → 类比：你"知道"的东西                             │
├─────────────────────────────────────────────────┤
│  Episodic Log（情景日志）                          │
│  → 按时间搜索                                     │
│  → 每段对话的摘要 + 关键话题                        │
│  → 只增不改（发生过的事不会变）                       │
│  → 类比：你"经历过"的东西                           │
└─────────────────────────────────────────────────┘
```

**为什么是三层不是一层？** 因为有些信息你每次都需要（用户名），有些你按意义搜（"用户喜欢什么编辑器"），有些你按时间搜（"上周聊了什么"）。把它们混在一起，要么浪费 token 全加载，要么搜索时错过关键信息。

---

## 写入管线（核心 IP）

当用户说了一段话，Nervon 做什么：

```
用户对话
   │
   ▼
[EXTRACT] LLM 提取原子事实
   │      "用户升职了" "用户换了编辑器"
   │
   ├──────────────────┐
   ▼                  ▼
[COMPARE]          [SUMMARIZE]
  每个事实搜索        生成对话摘要
  现有记忆           + 提取关键话题
  LLM 决定：
  ADD / UPDATE /      │
  DELETE / NOOP       │
   │                  │
   ▼                  ▼
 Semantic Store    Episodic Log
 (事实库)           (日志)
```

关键：**UPDATE 不是覆盖，是"退休"旧记忆 + 创建新记忆**。历史永远在。

---

## vs Mem0（最大竞品，$150M 估值）

| | Mem0 | Nervon |
|---|---|---|
| 架构 | 一层（扁平向量库） | 三层（按访问模式分） |
| 时间处理 | 没有。覆盖就没了 | 时间版本控制，旧记忆标"已过期" |
| 遗忘 | 没有。记忆无限增长 | 信心衰减（v1.1），越久没用越不重要 |
| 输出 | 原始 JSON | `get_context()` 直接给 prompt 用 |
| 锁定 | 有云服务依赖 | 纯 Python + SQLite，完全本地 |
| 多 agent | 有 scope，但无冲突解决 | v2 专攻（36.9% 多agent失败率的核心问题） |

**Mem0 用 LLM 在写入时做一件事（提取+决策）。我们在写入、检索、更新、维护全程用 LLM 推理。** 这是架构级别的差异，不是功能级别的。

---

## 已完成 ✅

### 代码（1,262 行核心 + 1,050 行测试）

| 模块 | 文件 | 状态 |
|-------|------|------|
| **数据模型** | `nervon/models.py` — Memory, Episode, WorkingMemoryBlock, MemorySearchResult | ✅ |
| **存储层** | `nervon/storage/sqlite.py` — SQLite 后端，向量搜索（numpy cosine），完整 CRUD | ✅ |
| **提取管线** | `nervon/pipeline/extract.py` — LLM 事实提取 | ✅ |
| **对比管线** | `nervon/pipeline/compare.py` — LLM ADD/UPDATE/DELETE 决策 | ✅ |
| **摘要管线** | `nervon/pipeline/summarize.py` — LLM 对话摘要 | ✅ |
| **嵌入** | `nervon/pipeline/embeddings.py` — 通过 litellm 支持任意模型 | ✅ |
| **搜索** | `nervon/retrieval/search.py` — 语义搜索 | ✅ |
| **上下文组装** | `nervon/retrieval/context.py` — 三层合并输出 | ✅ |
| **客户端** | `nervon/client.py` — MemoryClient 公开 API | ✅ |

### 测试（38/38 通过）

- 数据模型测试（9）
- 存储层测试（含 CRUD、向量搜索、时间过滤）
- 管线测试（提取、对比、摘要、嵌入）（15）
- 检索测试（5）
- 客户端集成测试（9）
- **真实 LLM 端到端测试**（Anthropic claude-3-haiku + 本地 sentence-transformers 嵌入）

### 发布

| 渠道 | 状态 | 链接 |
|------|------|------|
| **PyPI** | ✅ `pip install nervon` | https://pypi.org/project/nervon/0.1.0/ |
| **GitHub** | ✅ public repo | https://github.com/unicornnoway/nervon |
| **NPM** | — | Python only（暂时） |

### 真实验证

- ✅ 用 Anthropic API 跑通完整 add → search → update → get_context 流程
- ✅ 正确处理"用户搬家"场景（NYC → LA，旧记忆被 retire）
- ✅ 语义搜索返回正确排序
- ✅ Episode 自动生成对话摘要
- ✅ 把 OpenClaw 现有记忆（751条 memories + 23 episodes）迁移到 Nervon 数据库
- ✅ 创建了 OpenClaw test-agent（使用 Nervon 作为记忆后端）

### 其他产出

- ✅ 11 章完整架构文档（含竞品分析、认知科学理论、性能目标）
- ✅ Crypto × AI 破圈 Playbook
- ✅ X Thread 草稿（10条推文）
- ✅ Karpathy 风格配图（4张 matplotlib 生成）
- ✅ twikit X 数据抓取方案（免费，已 patch 可用）
- ✅ Mem0 源码深度分析（main.py, prompts.py, base.py）

---

## 未完成 & 待做 📋

### 高优先（本周）

| 任务 | 优先级 | 预计时间 | 说明 |
|------|--------|---------|------|
| **Nervon-memory skill** | 🔴 P0 | 2h | 让 test-agent 真正用 Nervon 做记忆读写 |
| **绑定 test-agent 到 Telegram** | 🔴 P0 | 30min | 才能真实对话测试 |
| **压力测试** | 🔴 P0 | 3h | 50+轮对话、矛盾处理、信息密度、模糊更新 |
| **发 X Thread** | 🟡 P1 | 1h | 配图已做好，文案已写好，就差发 |
| **修搜索质量** | 🟡 P1 | 2h | 迁移数据的中英混合搜索偏差需要调 |
| **Embedding 模型升级** | 🟡 P1 | 1h | MiniLM 太小，换多语言模型或 API 嵌入 |

### 中优先（本月）

| 任务 | 说明 |
|------|------|
| **LOCOMO benchmark** | 学术标准测试，跑分对标 Mem0 |
| **遗忘机制（v1.1）** | Ebbinghaus 衰减 + 使用频率加权 |
| **Demo chatbot** | CLI 聊天机器人 showcase，录屏发 X |
| **README 加 badge** | PyPI 版本、测试状态、license |
| **CI/CD** | GitHub Actions 自动测试 + 发版 |

### 远期（v2+）

| 任务 | 说明 |
|------|------|
| **多 Agent 共享记忆** | 核心差异化，36.9% 多agent失败率来自协调问题 |
| **LLM 重排序检索** | 向量粗筛 + LLM 精排 |
| **PostgreSQL 后端** | 生产级存储，pgvector |
| **Source tracking** | 记忆来源追踪 |
| **Cross-concept 抽象** | 跨概念归纳总结 |
| **Graph 关系层** | 评估是否需要图数据库 |
| **Hosted API** | 付费 SaaS（v3） |

---

## 技术栈

```
Language:     Python 3.11+
LLM:          litellm（支持 OpenAI / Anthropic / Ollama / Azure / 任意）
Embedding:    可插拔（sentence-transformers 本地 / OpenAI / 任意）
Storage:      SQLite（v1）→ PostgreSQL（v2）
Vector:       numpy cosine similarity（v1）→ pgvector / HNSW（v2）
Testing:      pytest（38 tests）
Build:        setuptools
License:      MIT
```

---

## 项目数据

```
代码行数:      1,262 行（核心）+ 1,050 行（测试）= 2,312 行
Python 文件:   16 个核心 + 7 个测试
测试覆盖:      38/38 通过
PyPI 下载:     v0.1.0 发布中
GitHub Stars:  0（刚发布）
迁移数据:      751 memories + 23 episodes（从 OpenClaw 导入）
Git 提交:      10 commits on main
开发时间:      ~8 小时（从架构设计到 PyPI 发布）
成本:          ~$0.50（LLM API 调用）
```

---

## 市场定位

```
赛道:         AI Agent Infrastructure → Memory Layer
对标:         Mem0 ($150M), Zep, Letta, HydraDB
差异化:       Reasoning throughout lifecycle (not just write-time)
              3-tier architecture (not flat)
              Temporal versioning (not overwrite)
              Open-source + local-first (not SaaS-locked)
商业模式:      Open Core — SDK 免费，Hosted API 收费（v3+）
```

---

## 战略路线图

```
NOW (Week 1)
├── ✅ V1 代码完成 + PyPI 发布
├── ✅ 真实 LLM 验证通过
├── ✅ OpenClaw 记忆迁移
├── 🔲 Test agent 完整集成
├── 🔲 压力测试 + 修搜索质量
└── 🔲 发第一条 X Thread

MONTH 1
├── LOCOMO benchmark 跑分
├── 遗忘机制 (v1.1)
├── Demo chatbot + 录屏
├── 每周 X Thread
├── 报名第一个 hackathon
└── 进入 2-3 个 Crypto AI 社区

MONTH 2-3
├── 多 Agent 共享记忆 (v2)
├── PostgreSQL 后端
├── LLM 重排序检索
├── 争取第一批外部用户
└── 考虑申请 grants

MONTH 4-6
├── Hosted API (v3)
├── 多模态记忆
├── 跨记忆推理
└── 如果 traction 好 → 考虑融资
```

---

## 关键风险

| 风险 | 等级 | 缓解 |
|------|------|------|
| Mem0 抄我们的差异化特性 | 🟡 中 | 先发优势 + 持续迭代速度 |
| LLM 成本限制用户采用 | 🟡 中 | 支持本地模型（Ollama），用便宜模型（Haiku） |
| 搜索质量不如专业向量数据库 | 🟡 中 | v2 接入 pgvector/HNSW |
| 一个人做不完 | 🔴 高 | 优先 MVP，hackathon 找合作者 |
| 市场不认 "reasoning-native" 概念 | 🟡 中 | 用 benchmark 数据说话 |

---

## 总结

**Nervon 不是"又一个 AI 记忆库"。** 

它是对"AI 怎么记忆"这个问题的架构级重新思考。在一个 $150M 估值的竞品只做了向量搜索 + LLM 提取的市场里，我们做了三层架构、时间版本控制、全链路 LLM 推理、和 prompt-ready 输出。

8 小时。2,312 行代码。38 个测试全过。已上 PyPI。真实 LLM 验证通过。

**下一步：让它在真实场景里跑起来，用数据证明它比 Mem0 好。**

---

*Built by Russ × Unicorn 🦄 | 2026-03-19*
