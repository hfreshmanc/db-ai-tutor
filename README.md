# DB-AI Tutor（数据库 AI 助教）

面向数据库课程场景的全栈 AI 助教：支持多轮对话、知识库 **RAG（检索增强生成）**，以及 Markdown 格式的专业回答展示。

用户可上传 `.txt` / `.md` / `.sql` 等学习资料，系统将其向量化存入 **Pinecone**；对话时从向量库检索相关片段，注入 **通义千问（DashScope）** 的 Prompt，使回答优先依据上传资料。

---

## 技术架构

```
┌─────────────┐     /api/*      ┌──────────────┐     DashScope     ┌─────────────┐
│  Vue 3 前端  │ ──────────────► │ FastAPI 后端  │ ───────────────► │  通义千问    │
│  Vite :5173 │   Vite 代理     │ Uvicorn :3001 │                  │ qwen3.6-plus│
└─────────────┘                 └──────┬───────┘                  └─────────────┘
                                       │
                         RAG 建库/检索  │
                                       ▼
                              ┌────────────────┐
                              │ Jina Embeddings │
                              │  + Pinecone     │
                              └────────────────┘
```

| 层级 | 技术 |
|------|------|
| 前端 | Vue 3、TypeScript、Vite 6、Tailwind CSS 4、Lucide Icons、markdown-it |
| 后端 | Python 3.9+、FastAPI、Uvicorn、httpx |
| 对话模型 | 阿里云 DashScope（默认 `qwen3.6-plus`，自动选择 multimodal / text 接口） |
| RAG | Jina Embeddings v3 → Pinecone 向量检索 |
| 开发联调 | concurrently 一键启动；Vite 代理 `/api` 至后端 |

---

## 主要功能

- **智能问答**：数据库原理、SQL 编写、范式、性能优化等
- **知识库 RAG**：上传资料 → 切块 → 向量化 → Pinecone 建库；对话时 Top-K 检索并优先依据资料回答
- **Prompt 约束**：RAG 模式下 system + user 双层提示，要求模型优先引用检索片段、禁止编造讲义内容
- **Markdown 渲染**：助手回复支持标题、列表、代码块等格式展示
- **API Key 灵活配置**：服务端 `.env` 或前端侧边栏填写 DashScope Key

---

## 环境要求

- **Node.js** 18+
- **Python** 3.9+
- 有效 API Key：**DashScope**（对话）、**Jina**（嵌入）、**Pinecone**（向量库，RAG 必需）

---

## 快速开始

### 1. 克隆并安装依赖

```bash
# 前端
npm install

# 后端（建议使用虚拟环境）
pip install -r requirements.txt
```

### 2. 配置环境变量

复制模板并填入你的 Key：

```bash
cp .env.example .env
```

`.env` 最小配置示例：

```env
# 对话（必需）
DASHSCOPE_API_KEY=sk-xxx
QWEN_MODEL=qwen3.6-plus

# RAG（启用知识库时必需）
JINA_API_KEY=jina_xxx
PINECONE_API_KEY=pcsk_xxx
PINECONE_INDEX=db-ai-tutor
JINA_EMBEDDING_DIMENSIONS=768
```

> **注意**：Pinecone 索引创建时的 `dimension` 必须与 `JINA_EMBEDDING_DIMENSIONS` 一致（默认 768）。  
> `qwen3.6-plus` 使用 **multimodal-generation** 接口；若改用 `qwen-plus` 等纯文本模型，后端会自动切换为 **text-generation** 接口。

### 3. 启动开发服务

```bash
npm run dev
```

| 服务 | 地址 |
|------|------|
| 前端 | http://localhost:5173 |
| 后端 API | http://127.0.0.1:3001 |
| 健康检查 | http://127.0.0.1:3001/api/health |

前端通过 Vite 将 `/api` 代理到 `127.0.0.1:3001`（Windows 下避免 `localhost` 解析到 IPv6 `::1` 导致连接失败）。

### 4. 验证链路（可选）

```bash
# Jina → Pinecone 建库链路
python test_jina_embeddings.py

# DashScope 对话链路
python test_qwen_chat.py
```

---

## 使用说明

1. 浏览器打开 http://localhost:5173
2. 侧边栏 **上传学习资料**（`.txt` / `.md` / `.sql`）
3. 确认 **RAG 模式** 已开启（上传成功后默认开启）
4. 在聊天框提问；与资料相关的问题会基于检索结果回答
5. 可在 **模型设置** 中填写个人 DashScope Key（留空则使用服务端 `.env`）

---

## RAG 流程简述

**建库（上传时，一次性）**

1. 前端读取文件文本 → `POST /api/rag/index`
2. **Parent-Child 切块**（默认）：parent ~1000 字符保留完整知识点，child ~250 字符用于检索
3. 仅对 **child** 调用 Jina `retrieval.passage` 向量化 → Pinecone upsert（metadata 含 `parentContext`）
4. 可通过 `RAG_CHUNK_STRATEGY=fixed` 回退为固定长度切块

**对话（每次提问，RAG 开启时）**

1. 用户问题 → Jina `retrieval.query` 向量化 → Pinecone Top-K 检索（命中 child 向量）
2. 按 `parentId` 去重，**返回 parent context** 给大模型（附带命中子片段摘要）
3. 检索片段写入 Prompt → 通义千问生成回答 → 前端 Markdown 渲染

---

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/api/health` | 服务状态、模型与 RAG 管线信息 |
| `POST` | `/api/chat` | 多轮对话（可选 RAG） |
| `POST` | `/api/rag/index` | 上传文本建库 |
| `POST` | `/api/rag/clear` | 清空 Pinecone 索引 |

---

## 环境变量参考

### 必需（完整 RAG + 对话）

| 变量 | 说明 |
|------|------|
| `DASHSCOPE_API_KEY` | 通义千问 API Key |
| `JINA_API_KEY` | Jina Embeddings API Key |
| `PINECONE_API_KEY` | Pinecone API Key |
| `PINECONE_INDEX` | Pinecone 索引名称 |

### 常用可选

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `QWEN_MODEL` | `qwen3.6-plus` | 对话模型 |
| `DASHSCOPE_API_URL` | 按模型自动选择 | 手动覆盖 DashScope 接口地址 |
| `JINA_EMBEDDING_MODEL` | `jina-embeddings-v3` | 嵌入模型 |
| `JINA_EMBEDDING_DIMENSIONS` | `768` | 向量维度（须与 Pinecone 索引一致） |
| `JINA_TASK_PASSAGE` | `retrieval.passage` | 建库 embedding task |
| `JINA_TASK_QUERY` | `retrieval.query` | 检索 embedding task |
| `RAG_CHUNK_STRATEGY` | `parent_child` | 切块策略：`parent_child` 或 `fixed` |
| `RAG_PARENT_SIZE` | `1000` | Parent 块大小（字符） |
| `RAG_PARENT_OVERLAP` | `200` | Parent 块重叠 |
| `RAG_CHILD_SIZE` | `250` | Child 块大小（用于 embedding） |
| `RAG_CHILD_OVERLAP` | `50` | Child 块重叠 |
| `RAG_CHUNK_SIZE` | `800` | 仅 `fixed` 策略时使用 |
| `RAG_CHUNK_OVERLAP` | `80` | 仅 `fixed` 策略时使用 |
| `RAG_TOP_K` | `3` | 检索返回片段数 |
| `JINA_EMBED_BATCH` | `64` | Jina 单次 embedding 批大小 |
| `JINA_HTTP_TIMEOUT_SECONDS` | `300` | Jina 请求超时（秒） |

### 开发环境（Vite）

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `DEV_API_PROXY_TARGET` | `http://127.0.0.1:3001` | 前端 API 代理目标 |
| `DEV_API_PROXY_TIMEOUT_MS` | `900000` | 代理超时（RAG 建库较慢） |
| `DISABLE_HMR` | — | 设为 `true` 关闭热更新 |

---

## 项目结构

```
db-ai-tutor/
├── src/
│   ├── App.vue                 # 主界面（聊天、RAG 开关、资料上传）
│   ├── components/
│   │   └── MarkdownMessage.vue # 助手回复 Markdown 渲染
│   └── index.css               # Tailwind + prose 样式
├── main.py                     # FastAPI 入口（对话 + RAG API）
├── jina_pinecone_index.py      # Jina ↔ Pinecone 建库/检索
├── rag_service.py              # RAG 服务（语义分块等扩展逻辑）
├── test_jina_embeddings.py     # Jina → Pinecone 冒烟测试
├── test_qwen_chat.py           # DashScope 对话冒烟测试
├── vite.config.ts              # 前端构建与开发代理
├── requirements.txt            # Python 依赖
└── package.json                # Node 脚本与前端依赖
```

---

## 常用命令

```bash
npm run dev          # 前后端并行开发
npm run dev:frontend # 仅前端 (5173)
npm run dev:backend  # 仅后端 (3001)
npm run build        # 构建前端到 dist/
npm run start        # 生产模式启动后端（托管 dist/ 静态文件）
npm run lint         # Vue/TS 类型检查
```

---

## 生产部署

```bash
npm run build
npm run start
```

构建后 FastAPI 会自动挂载 `dist/` 目录，单进程即可同时提供 API 与前端静态资源（默认端口 `3001`）。

---

## 常见问题

**Q: 对话返回 `url error` / `InvalidParameter`？**  
A: 检查 `QWEN_MODEL` 与接口是否匹配。`qwen3.6-plus` 需 multimodal 接口，后端已自动处理；若手动设置了 `DASHSCOPE_API_URL`，请确保与模型类型一致。

**Q: RAG 建库前端超时？**  
A: 开发环境下调大 `DEV_API_PROXY_TIMEOUT_MS`；后端 Jina 超时调 `JINA_HTTP_TIMEOUT_SECONDS`。

**Q: Windows 下 API 代理 ECONNREFUSED？**  
A: 代理目标使用 `127.0.0.1` 而非 `localhost`（已在 `vite.config.ts` 默认配置）。

**Q: Pinecone upsert 维度错误？**  
A: 索引 `dimension` 必须与 `JINA_EMBEDDING_DIMENSIONS` 一致；修改维度后需清空索引并重新上传资料。

---

## License

Private / 学习项目用途。
