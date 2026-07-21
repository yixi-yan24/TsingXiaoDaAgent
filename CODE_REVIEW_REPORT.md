# 🔬 TsingXiaoDaAgent 代码审查报告

> **项目**：清华大学辅修专业规划助手（Tsinghua Minor Advisor）
> **审查日期**：2026-07-21
> **代码版本**：master 分支（09d2125）

---

## 一、项目概述

基于 DeepSeek API 构建的智能 Agent，为清华本科生提供辅修专业咨询与修读规划服务。兼容 OpenAI `/v1/chat/completions` 格式，支持 CLI 交互和 HTTP API 两种模式。

### 技术栈

| 组件 | 技术选型 |
|------|----------|
| LLM | DeepSeek API（OpenAI 兼容） |
| Web 框架 | FastAPI + Uvicorn |
| 词嵌入 | sentence-transformers / text2vec-base-chinese |
| 数据存储 | Markdown 解析 + JSON 缓存 |
| 部署 | Docker + docker-compose |

---

## 二、架构总览

```
TsingXiaoDaAgent/
├── agent/                     # Agent 核心
│   ├── core.py                # 会话管理、LLM 调用、ReAct 工具调度
│   ├── data_loader.py         # 培养方案 Markdown 解析 → 结构化数据
│   ├── embedding.py           # 词嵌入引擎：语义搜索
│   ├── memory.py              # 短期记忆（对话历史）& 长期记忆（辅修数据库）
│   ├── tools.py               # 工具集：搜索、详情、资格检查
│   ├── course_catalog.py      # 已整理课程资料的本地检索目录
│   ├── course_graph.py        # 课程 DAG 构建 & 拓扑排序
│   ├── llm_client.py          # 统一的 LLM 调用 + 重试
│   ├── multi_agent.py         # 多 Agent 协同（档案分析/搜索推荐/计划审核）
│   ├── prompts.py             # 系统提示词模板
│   └── planner.py             # 双通道修读计划生成
├── api/
│   └── main.py                # FastAPI 服务（OpenAI 兼容格式）
├── data/                      # 解析缓存
├── Dockerfile & docker-compose.yml
├── run.py                     # 统一入口
├── curated_courses.json       # 已整理的课程资料
└── 本科辅修培养方案2026版.md
```

### 核心工作流

```
用户提问 → AgentSession.process_message()
              ↓
         _call_llm() → 构建 tool-augmented prompt
              ↓
         DeepSeek API → 返回 THOUGHT/ACTION/PARAMS 或最终回答
              ↓
     ┌─ 有工具调用? → _execute_tool() → 结果回填 STM → 递归 _call_llm()
     │                                          (最多3次)
     └─ 无工具调用? → _clean_response() → 返回用户
```

---

## 三、分模块评审

### 3.1 `core.py` — Agent 核心

**职责**：会话管理、ReAct 循环、工具调度

| 评价 | 说明 |
|------|------|
| ✅ 好 | `MinorAdvisorAgent`（工厂）与 `AgentSession`（实例）分离，设计合理 |
| ✅ 好 | 工具调用上限 3 次防止无限循环 |
| ✅ 好 | `_clean_response` 移除内部 `THOUGHT:` 行，避免暴露推理过程 |
| ❌ 问题 | **system prompt 中 tool block 被重复追加**：`_call_llm` 每次递归都在 system 消息末尾拼接完整的 tool block，3 次工具调用后 system prompt 包含 4 份副本，token 消耗线性增长 |
| ❌ 问题 | **`_execute_tool` 每次重建 tool_map**：应使用 `getattr` 动态分发或缓存映射表 |
| ❌ 问题 | **`_clean_response` 存在 O(n²) 性能问题**：内层 `any()` 每次循环都扫描全部行 |
| ❌ 问题 | **prompt-based tool calling 脆弱**：依赖正则解析 `ACTION:`/`PARAMS:`，LLM 格式偏差即失败。DeepSeek 已支持原生 `tool_choice` 但未使用 |

> **改进建议**：在首次调用时注入 tool block，后续递归使用原始 system prompt + 独立的 tool result 消息。长期迁移至 DeepSeek 原生 function calling。

---

### 3.2 `llm_client.py` — LLM 调用客户端

**职责**：统一的 API 调用、超时控制、瞬时失败重试

| 评价 | 说明 |
|------|------|
| ✅ 好 | 重试间隔递增（0.5s × attempt） |
| ✅ 好 | 空响应检测 |
| ❌ 问题 | **异常分类过于宽泛**：`KeyError`、`IndexError`、`TypeError`、`ValueError` 被当作网络错误重试，这些是逻辑错误，重试无意义且浪费 API 额度 |
| ❌ 问题 | **模型名硬编码**：`"model": "deepseek-chat"` 写死在函数内，无法切换其他兼容 API |
| ❌ 问题 | **无速率限制**：高并发下可能触发上游限流 |

> **改进建议**：
> ```python
> # 区分可重试错误与不可重试错误
> RETRYABLE = (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPStatusError)
> NON_RETRYABLE = (KeyError, IndexError, TypeError, ValueError)
> ```

---

### 3.3 `memory.py` — 记忆管理

**职责**：短期记忆（对话历史）和长期记忆（辅修数据库）

| 评价 | 说明 |
|------|------|
| ✅ 好 | `ShortTermMemory.add()` 自动截断超长消息和轮次 |
| ✅ 好 | `to_llm_format()` 将 tool 消息转为 user 角色，兼容受限 API |
| ❌ 问题 | **轮次计算不准确**：`max_turns * 2` 假设每轮只有 user+assistant，忽略了 tool 消息，实际保留轮次少于预期 |

> **改进建议**：按 user 消息计数，或基于 token 计数而非消息条数进行截断。

---

### 3.4 `tools.py` — 工具集

**职责**：9 个工具函数 + 工具描述声明

| 评价 | 说明 |
|------|------|
| ✅ 好 | 工具描述与实现分离，声明式元数据清晰 |
| ✅ 好 | `_format_course_detail` 关注点分离良好 |
| ❌ 问题 | **`semantic_search` 重复加载数据**：调用 `load_minors()` 重新读盘，完全忽略传入的 `self.ltm` |
| ❌ 问题 | **`multi_agent_search` 每次新建 MultiAgentSystem**：应复用实例，避免重复创建 3 个 SpecialistAgent |
| ❌ 问题 | **`list_minors` 无分页**：44 个专业全量返回，浪费上下文窗口 |

> **改进建议**：`semantic_search` 直接使用 `self.ltm.minors`；`multi_agent_search` 在 `Tools.__init__` 中创建并缓存 MultiAgentSystem 实例。

---

### 3.5 `data_loader.py` — 数据解析

**职责**：Markdown 培养方案 → 结构化 `MinorProgram` 对象 + JSON 缓存

| 评价 | 说明 |
|------|------|
| ✅ 好 | 解析 + 缓存两层机制，兼顾灵活性和性能 |
| ✅ 好 | `force_reload` 参数设计合理 |
| ❌ 问题 | **解析高度依赖 Markdown 格式**：`##` 标题结构变化即导致解析失败，无格式校验 |
| ❌ 问题 | **`_guess_department` 只回溯 10 行**：院系标题距离较远时匹配失败 |
| ❌ 问题 | **`_extract_metadata` 总学分正则过于宽泛**：`text[:200]` 中 `([\d.]+)\s*学分` 可能匹配到无关数字 |

---

### 3.6 `planner.py` — 修读计划生成

**职责**：双通道规划（算法拓扑排序 + LLM 增强 + Multi-Agent 审核）

| 评价 | 说明 |
|------|------|
| ✅ 好 | **三级规划策略是本项目最大的创新点**：算法保证科学性，LLM 增强可读性，MultiAgent 审核可靠性 |
| ✅ 好 | 输出包含咨询电话、接纳人数等实用元数据 |
| ❌ 问题 | **异常信息直接暴露给用户**：`f"生成计划时出错: {e}"` 可能泄露内部信息 |

> **改进建议**：生产环境使用通用错误消息，将详细异常记录到日志。

---

### 3.7 `multi_agent.py` — 多 Agent 协同

**职责**：Profile Analyzer → Search Recommender → Plan Verifier 三级流水线

| 评价 | 说明 |
|------|------|
| ✅ 好 | `SpecialistAgent` 基类简洁，各 Agent 职责单一 |
| ❌ 问题 | **`search_recommendations` 硬编码 `[:20]` 限制**：后半部分辅修专业永远不被 MultiAgent 推荐 |
| ❌ 问题 | **`_call_llm` 丢弃 `tool_name` 字段**：维护陷阱，虽当前不触发但未来可能出 bug |

> **改进建议**：用 `semantic_search` 对全部 44 个专业预筛选 Top-N，再传给 LLM 分析。

---

### 3.8 `course_graph.py` — 课程图/拓扑排序

**职责**：课程 DAG 构建、拓扑排序、学期规划

| 评价 | 说明 |
|------|------|
| ✅ 好 | DAG + 拓扑排序实现完整，考虑了开课学期约束 |
| ❌ 问题 | **参数命名误导**：`parse_courses_from_table(html_table)` 实则接收 Markdown |
| ❌ 问题 | **拓扑排序死锁处理破坏先修约束**：无法排课时强行取一个课程 |
| ❌ 问题 | **`year_labels` 变量定义了但未使用**：死代码 |
| ❌ 问题 | **不支持"春/秋"双学期课程**：`"春秋"` 标记的课程在两学期都应可排 |

---

### 3.9 `embedding.py` — 词嵌入

**职责**：中文语义搜索，基于 text2vec-base-chinese

| 评价 | 说明 |
|------|------|
| ✅ 好 | 全局单例模型加载，避免重复初始化 |
| ✅ 好 | 嵌入索引缓存到磁盘，支持增量更新检测 |
| ✅ 好 | `HF_ENDPOINT` 默认使用国内镜像 |
| ❌ 问题 | **pickle 存储有安全隐患**：被篡改的缓存文件可执行任意代码 |
| ❌ 问题 | **未使用 `normalize_embeddings` 的一致性**：`build_index` 中 normalize 了，但在 `semantic_search` 中又 normalize 了一次（重复计算） |

> **改进建议**：用 `numpy.save` / `numpy.load` 替代 pickle；在 `build_index` 时 normalize 后直接用 dot product 等价于 cosine similarity。

---

### 3.10 `api/main.py` — API 服务

**职责**：OpenAI 兼容的 HTTP API + Session 管理

| 评价 | 说明 |
|------|------|
| ✅ 好 | OpenAI 兼容格式实现完整 |
| ✅ 好 | LRU 淘汰的 session 管理（`OrderedDict` + `MAX_SESSIONS = 1000`） |
| ✅ 好 | 兼容 Page Assist 等不传 `user` 字段的客户端 |
| ❌ 问题 | **伪流式输出**：LLM 完整响应后才按 64 字符切片发送，首 token 延迟 = 完整响应延迟 |
| ❌ 问题 | **`_convert_openai_to_agent` 只取最后一条 user 消息**：丢失 system prompt 中的自定义指令 |
| ❌ 问题 | **全局单例 Agent**：`_agent = MinorAdvisorAgent(...)` 在模块加载时创建，并发场景下 session 创建正确但不够优雅 |
| ❌ 问题 | **路由函数是同步的**：`def chat_completions(...)` 而非 `async def`，FastAPI 会用线程池执行，可能阻塞 |

---

### 3.11 `course_catalog.py` — 课程目录

**职责**：curated_courses.json 的结构化检索

| 评价 | 说明 |
|------|------|
| ✅ 好 | `_normalize` 函数统一文本标准化 |
| ✅ 好 | 分层搜索策略（精确 ID → 名称 → 模糊匹配） |
| ❌ 问题 | **`if-elif` 评分链导致无法综合评分**：一旦匹配高优先级条件就停止，搜索结果排序维度单一 |

---

### 3.12 `prompts.py` — 提示词模板

| 评价 | 说明 |
|------|------|
| ✅ 好 | 提示词结构清晰，包含角色定义、工作流程、工具使用指导、重要规则、输出风格 |
| ✅ 好 | 强调了 semantic_search 优先于 keyword search 的场景区分 |
| ⚠️ 注意 | `PLANNER_PROMPT` 和 `QUERY_PROMPT` 在代码中仅部分使用（`QUERY_PROMPT` 似乎未在任何地方引用） |

---

## 四、综合风险清单

| # | 严重度 | 位置 | 问题 | 影响 |
|---|--------|------|------|------|
| 1 | 🔴 高 | `llm_client.py:36` | `KeyError/IndexError/TypeError/ValueError` 被当作网络错误重试 | 浪费 API 额度，掩盖真实 bug |
| 2 | 🔴 高 | `core.py:83-128` | system prompt 中 tool block 递归追加 | token 消耗线性增长 |
| 3 | 🟡 中 | `api/main.py:161-169` | 伪流式 SSE 输出 | 用户体验差，首字节延迟高 |
| 4 | 🟡 中 | `tools.py:157` | `semantic_search` 重复 `load_minors()` | 不必要的 IO，影响响应速度 |
| 5 | 🟡 中 | `multi_agent.py:99` | `ltm.minors[:20]` 硬编码截断 | 后半部分辅修永远不会被推荐 |
| 6 | 🟡 中 | `planner.py:96` | 异常信息直接返回给用户 | 信息泄露风险 |
| 7 | 🟡 中 | `core.py:148-160` | 正则解析 tool call 无容错 | LLM 输出格式偏差即失败 |
| 8 | 🟢 低 | `embedding.py:56` | pickle 加载安全隐患 | 仅本地使用风险较低 |
| 9 | 🟢 低 | `course_graph.py:182-193` | 拓扑排序强制破坏先修约束 | 极端情况产生不可靠规划 |
| 10 | 🟢 低 | `api/main.py` | 无 API 限流 | 高并发下触发上游限流 |

---

## 五、改进路线图

### 🔥 第一优先级（本周修复）

| # | 改进项 | 文件 | 预估工作量 |
|---|--------|------|-----------|
| 1 | 修复 tool block 重复追加 | `core.py:_call_llm` | 0.5h |
| 2 | 区分可重试与不可重试异常 | `llm_client.py:chat_completion` | 0.5h |

### 📌 第二优先级（本月完成）

| # | 改进项 | 文件 | 预估工作量 |
|---|--------|------|-----------|
| 3 | 接入 DeepSeek 原生 function calling 替代 prompt-based tool use | `core.py` | 4h |
| 4 | 实现真正的 SSE 流式输出 | `llm_client.py` + `api/main.py` | 3h |
| 5 | 修复 `semantic_search` 数据重复加载 | `tools.py` | 0.5h |
| 6 | MultiAgent 推荐突破 `[:20]` 限制 | `multi_agent.py` | 1h |
| 7 | 添加日志替代裸 print / 异常暴露 | 全局 | 2h |

### 🎯 第三优先级（季度规划）

| # | 改进项 | 说明 | 预估工作量 |
|---|--------|------|-----------|
| 8 | 单元测试覆盖 | 核心模块（data_loader, course_graph, memory, tools） | 8h |
| 9 | API rate limiter | 使用 `slowapi` 或自实现 token bucket | 2h |
| 10 | 替换 pickle 为 safetensors / numpy | 消除安全隐患 | 1h |
| 11 | `_convert_openai_to_agent` 保留 system 消息 | 支持 API 调用者自定义指令 | 2h |
| 12 | 异步化 API 路由 | `def → async def`，避免线程池阻塞 | 2h |

---

## 六、总体评价

| 维度 | 评分 | 说明 |
|------|------|------|
| **架构设计** | ⭐⭐⭐⭐ | 分层清晰，ReAct + 双通道规划 + MultiAgent 审核设计优秀 |
| **代码质量** | ⭐⭐⭐ | 可读性良好，注释适中，但部分细节粗糙 |
| **错误处理** | ⭐⭐ | 异常分类粗放，部分异常信息泄露 |
| **性能** | ⭐⭐⭐ | 基本合理，但存在重复数据加载、token 浪费等热点 |
| **安全性** | ⭐⭐⭐ | 整体安全，pickle 和异常暴露是低风险项 |
| **可维护性** | ⭐⭐⭐ | 模块职责清晰，但缺乏测试是最大短板 |
| **文档** | ⭐⭐⭐⭐ | README 清晰，API 文档完整，使用方法详细 |

**综合评分：7.0 / 10**

这是一个**定位准确、功能完整**的 AI Agent 应用。核心设计体现了对 LLM 应用架构的良好理解。代码风格务实，没有过度工程化。主要短板集中在 prompt 工程的 token 效率、异常处理的精细化、以及部分热路径的性能优化。这些问题不影响功能可用性，但在生产环境规模化部署前应优先解决第一优先级的两个问题。

---

> 📝 *本报告由 Claude Code 自动生成，基于对全部 15 个源文件的逐行审查。*
