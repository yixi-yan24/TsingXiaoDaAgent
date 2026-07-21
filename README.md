# Tsinghua Minor Advisor — 清华大学辅修专业规划助手

基于 DeepSeek API 构建的智能 Agent，为清华本科生提供辅修专业咨询与修读规划服务。兼容 OpenAI `/v1/chat/completions` 格式，支持流式输出。

---

## 快速开始

### 前置要求

- Python 3.10+
- DeepSeek API Key（或兼容 OpenAI 格式的其他 API）

### 本地运行

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 API Key（二选一）
export DEEPSEEK_API_KEY=sk-xxx          # 环境变量
# 或将 Key 写入 .config 文件（自动读取）

# 3. 启动 CLI 交互模式
python run.py

# 或启动 API 服务
python run.py --mode api --port 8000
```

### Docker 部署

```bash
# 构建镜像
docker build -t tsinghua-minor-advisor .

# 运行
docker run -d --name minor-advisor \
  -p 8000:8000 \
  -e DEEPSEEK_API_KEY=sk-xxx \
  -v ./data:/app/data \
  tsinghua-minor-advisor
```

或使用 docker compose：

```bash
export DEEPSEEK_API_KEY=sk-xxx
docker compose up -d
```

---

## API 文档

### OpenAI 兼容接口

**对话：** `POST /v1/chat/completions`

```json
{
  "model": "tsinghua-minor-advisor",
  "messages": [
    {"role": "user", "content": "我是计算机系大一学生，想辅修经济学"}
  ],
  "stream": true,
  "user": "会话ID（可选）"
}
```

可用任何 OpenAI SDK 调用：

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
resp = client.chat.completions.create(
    model="tsinghua-minor-advisor",
    messages=[{"role": "user", "content": "..."}],
    stream=True
)
for chunk in resp:
    print(chunk.choices[0].delta.content or "", end="")
```

**模型列表：** `GET /v1/models`

### 辅修数据接口

| 端点 | 说明 |
|------|------|
| `GET /minors` | 获取所有辅修专业列表 |
| `GET /minors/{名称}` | 获取某辅修详细信息 |

---

## 项目架构

```
TsingXiaoDaAgent/
├── agent/                  # Agent 核心
│   ├── core.py             # 会话管理、LLM 调用、ReAct 工具调度
│   ├── data_loader.py      # 长期记忆：解析辅养方案 → 结构化数据
│   ├── embedding.py        # 词嵌入引擎：语义搜索（sentence-transformers）
│   ├── memory.py           # 短期记忆（对话历史）& 长期记忆（辅修数据库）
│   ├── tools.py            # 工具集：搜索、详情、资格检查
│   ├── course_catalog.py   # 已整理课程资料的本地检索目录
│   ├── llm_client.py       # 统一的模型调用、超时与瞬时失败重试
│   ├── prompts.py          # 系统提示词模板
│   └── planner.py          # 修读计划生成
├── api/
│   └── main.py             # FastAPI 服务（OpenAI 兼容格式）
├── data/                   # 解析缓存（自动生成）
├── Dockerfile & docker-compose.yml
├── run.py                  # 统一入口
├── curated_courses.json    #课程介绍
└── 本科辅修培养方案2025版.md
```

### 设计要点

| 组件 | 说明 |
|------|------|
| **推理机制** | ReAct 模式：LLM 输出 `ACTION` 触发工具调用，结果回填后二次推理 |
| **短期记忆** | 每个会话独立的对话历史（最近 20 轮），以 `user` 字段区分 |
| **长期记忆** | 44 个辅修专业培养方案，以及 1000+ 门已整理的课程资料 |
| **长期记忆** | 44 个辅修专业的结构化数据（学分、课程、限制、联系方式等） |
| **词嵌入** | 基于 `shibing624/text2vec-base-chinese` 的语义搜索，余弦相似度排序 |
| **规划能力** | LLM 自主推理 + 专用 Planner 双通道，考虑先修关系、开课学期、学分均衡 |
| **工具集** | 	list_minors / search_minors / semantic_search / get_minor_detail / check_eligibility / multi_agent_search / search_courses /get_course_detail / list_minor_courses

---

## 使用示例

### CLI 模式

```
$ python run.py

清华大学辅修专业规划助手 v1.0
输入 'quit' 退出 | 'clear' 清空对话 | 'plan' 生成修读计划

你 > 我是计算机系大二学生，想辅修经济学
助手 > [根据你的情况推荐经济学辅修，并给出课程安排...]

你 > plan 机械工程 大二 计算机科学与技术
助手 > [生成按学期的详细修读计划...]
```

### API 模式

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tsinghua-minor-advisor",
    "messages": [{"role": "user", "content": "经济学辅修有哪些必修课？"}],
    "stream": true
  }'
```

---

## 配环境

依赖：`fastapi`、`uvicorn`、`httpx`、`pydantic`、`sentence-transformers`

```bash
pip install fastapi uvicorn httpx pydantic sentence-transformers
```

> 首次运行词嵌入功能时会自动下载模型（约 400MB），国内已配置 HF 镜像加速。也可通过环境变量 `HF_ENDPOINT` 自定义镜像源。

---

## License

MIT
