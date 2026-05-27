<script setup lang="ts">
import { ref, onMounted, nextTick, watch } from 'vue';
import { 
  Send, 
  Bot, 
  User, 
  Trash2, 
  Key, 
  FileText, 
  Loader2, 
  Settings, 
  ChevronRight,
  Database,
  Brain
} from 'lucide-vue-next';
import MarkdownMessage from './components/MarkdownMessage.vue';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  isRAG?: boolean;
  /** 错误信息等以纯文本展示，不做 Markdown 渲染 */
  plain?: boolean;
}

interface KBFile {
  name: string;
  size: number;
}

const input = ref('');
const messages = ref<Message[]>([]);
const isLoading = ref(false);
const isIndexing = ref(false);
const kbFiles = ref<KBFile[]>([]);
const useRAG = ref(false);
const dashscopeKey = ref('');
const showSettings = ref(false);
const scrollContainer = ref<HTMLElement | null>(null);

onMounted(() => {
  const savedDashscope = localStorage.getItem('dashscope_key');
  if (savedDashscope) dashscopeKey.value = savedDashscope;
});

watch(dashscopeKey, (val) => localStorage.setItem('dashscope_key', val));

/** 解析 FastAPI / 代理返回的错误正文，便于在界面与控制台看到后端原因 */
async function apiErrorMessage(resp: Response, tag: string): Promise<string> {
  const prefix = `[${tag}] HTTP ${resp.status}`;
  let body = '';
  try {
    body = await resp.text();
  } catch {
    return `${prefix}（无法读取响应体）`;
  }
  try {
    const j = JSON.parse(body) as { detail?: unknown };
    if (j?.detail !== undefined) {
      const d = j.detail;
      const msg = typeof d === 'string' ? d : JSON.stringify(d, null, 2);
      return `${prefix}\n${msg}`;
    }
  } catch {
    /* 非 JSON */
  }
  return body.trim() ? `${prefix}\n${body.slice(0, 4000)}` : prefix;
}

const scrollToBottom = () => {
  nextTick(() => {
    if (scrollContainer.value) {
      scrollContainer.value.scrollTop = scrollContainer.value.scrollHeight;
    }
  });
};

const handleSend = async () => {
  if (!input.value.trim() || isLoading.value) return;

  const userMsg = input.value;
  input.value = '';
  messages.value.push({ role: 'user', content: userMsg });
  isLoading.value = true;
  scrollToBottom();

  try {
    const history = messages.value.slice(0, -1).map(m => ({
      role: m.role === 'user' ? 'user' : 'assistant',
      parts: [{ text: m.content }]
    }));

    const response = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message: userMsg,
        history,
        systemInstruction:
          '你是一位专业的数据库助教，擅长数据库原理、SQL 和性能优化。',
        useRAG: useRAG.value,
        dashscopeKey: dashscopeKey.value,
        stream: false,
      }),
    });

    if (!response.ok) {
      const errText = await apiErrorMessage(response, 'api/chat');
      console.error(errText);
      messages.value.push({ role: 'assistant', content: errText, plain: true });
      return;
    }

    const data = await response.json();
    
    // 通义千问 DashScope 响应格式
    let reply = "";
    let plainReply = false;
    if (data.output?.choices?.[0]?.message?.content) {
      reply = data.output.choices[0].message.content;
    } else if (data.detail) {
      reply = typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail, null, 2);
      plainReply = true;
    }

    if (!reply) {
      const fallback = `接口返回 200 但无法解析内容：\n${JSON.stringify(data, null, 2).slice(0, 2000)}`;
      console.warn('[api/chat]', fallback);
      reply = fallback;
      plainReply = true;
    }

    messages.value.push({ 
      role: 'assistant', 
      content: reply,
      isRAG: useRAG.value,
      plain: plainReply,
    });
  } catch (error) {
    const msg = error instanceof Error ? error.message : String(error);
    console.error('[api/chat]', error);
    messages.value.push({ role: 'assistant', content: `请求异常：${msg}`, plain: true });
  } finally {
    isLoading.value = false;
    scrollToBottom();
  }
};

const handleFileUpload = async (event: Event) => {
  const target = event.target as HTMLInputElement;
  if (!target.files?.length) return;

  const file = target.files[0];
  isIndexing.value = true;
  
  try {
    const text = await file.text();
    const resp = await fetch('/api/rag/index', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text,
        filename: file.name,
      }),
    });
    
    if (!resp.ok) {
      const errText = await apiErrorMessage(resp, 'api/rag/index');
      console.error(errText);
      alert(errText);
      return;
    }

    const data = await resp.json();
    kbFiles.value.push({ name: file.name, size: file.size });
    useRAG.value = true;
    alert(`已写入 Pinecone：${data.count ?? 0} 条向量（索引 ${data.pinecone_index ?? ''}）`);
  } catch (error) {
    const msg = error instanceof Error ? error.message : String(error);
    console.error('[api/rag/index]', error);
    alert(`文件处理失败：${msg}`);
  } finally {
    isIndexing.value = false;
  }
};

const handleClearKB = async () => {
  await fetch('/api/rag/clear', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({}),
  });
  kbFiles.value = [];
  useRAG.value = false;
};
</script>

<template>
  <div class="flex h-screen bg-slate-50 text-slate-900 font-sans">
    <!-- Sidebar -->
    <div class="w-80 bg-white border-r border-slate-200 flex flex-col p-6 space-y-8">
      <div class="flex items-center space-x-3">
        <div class="p-2 bg-indigo-600 rounded-lg text-white">
          <Database class="w-6 h-6" />
        </div>
        <h1 class="text-xl font-bold tracking-tight">DB-AI Tutor</h1>
      </div>

      <div class="flex-1 space-y-6">
        <div>
          <label class="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-3 block">知识库 (RAG)</label>
          <div class="space-y-2">
            <div 
              v-for="file in kbFiles" 
              :key="file.name"
              class="flex items-center justify-between p-3 bg-slate-50 rounded-lg border border-slate-100 group"
            >
              <div class="flex items-center space-x-2">
                <FileText class="w-4 h-4 text-indigo-500" />
                <span class="text-sm font-medium truncate w-32">{{ file.name }}</span>
              </div>
              <button @click="handleClearKB" class="text-slate-300 hover:text-red-500 opacity-0 group-hover:opacity-100 transition-opacity">
                <Trash2 class="w-4 h-4" />
              </button>
            </div>

            <div v-if="kbFiles.length === 0" class="text-sm text-slate-400 italic p-3 text-center border border-dashed border-slate-200 rounded-lg">
              暂无文档
            </div>

            <label class="flex items-center justify-center space-x-2 p-3 bg-white border border-indigo-200 text-indigo-600 rounded-lg cursor-pointer hover:bg-indigo-50 transition-colors mt-4">
              <input type="file" @change="handleFileUpload" class="hidden" accept=".txt,.md,.sql" />
              <Loader2 v-if="isIndexing" class="w-4 h-4 animate-spin" />
              <FileText v-else class="w-4 h-4" />
              <span class="text-sm font-semibold">上传学习资料</span>
            </label>
          </div>
        </div>

        <div class="p-4 bg-indigo-50 rounded-xl space-y-3">
          <div class="flex items-center justify-between">
            <div class="flex items-center space-x-2">
              <Brain class="w-4 h-4 text-indigo-600" />
              <span class="text-sm font-semibold text-indigo-900">RAG 模式</span>
            </div>
            <button 
              @click="useRAG = !useRAG"
              :class="[
                'w-10 h-5 rounded-full transition-colors relative',
                useRAG ? 'bg-indigo-600' : 'bg-slate-300'
              ]"
            >
              <div :class="['w-3 hole h-3 bg-white rounded-full absolute top-1 transition-all', useRAG ? 'right-1' : 'left-1']" />
            </button>
          </div>
          <p class="text-xs text-indigo-700 leading-relaxed">
            启用后 AI 将基于你上传的资料进行回答。
          </p>
        </div>
      </div>

      <div class="space-y-4">
        <button 
          @click="showSettings = !showSettings"
          class="flex items-center justify-between w-full p-3 text-slate-500 hover:bg-slate-50 rounded-lg transition-colors"
        >
          <div class="flex items-center space-x-2">
            <Settings class="w-4 h-4" />
            <span class="text-sm font-medium">模型设置</span>
          </div>
          <ChevronRight :class="['w-4 h-4 transition-transform', showSettings ? 'rotate-90' : '']" />
        </button>

        <div v-show="showSettings" class="space-y-4 p-3 bg-slate-50 rounded-lg">
          <div>
            <label class="text-[10px] font-bold text-slate-400 uppercase mb-1 block">DashScope (通义千问) Key</label>
            <input 
              v-model="dashscopeKey"
              type="password"
              placeholder="留空则使用服务端 .env 中的 DASHSCOPE_API_KEY"
              class="w-full p-2 text-xs border border-slate-200 rounded focus:ring-1 focus:ring-indigo-500 outline-none"
            />
          </div>
        </div>
      </div>
    </div>

    <!-- Main Chat -->
    <div class="flex-1 flex flex-col relative">
      <div 
        ref="scrollContainer"
        class="flex-1 overflow-y-auto p-12 space-y-8"
      >
        <div v-if="messages.length === 0" class="flex flex-col items-center justify-center h-full text-center space-y-4 max-w-md mx-auto">
          <div class="w-16 h-16 bg-indigo-100 text-indigo-600 rounded-2xl flex items-center justify-center mb-4">
            <Bot class="w-8 h-8" />
          </div>
          <h2 class="text-2xl font-bold text-slate-900">你好，我是你的数据库助教</h2>
          <p class="text-slate-500 leading-relaxed">我可以帮你解答 SQL 问题、解释范式化，或者帮你设计复杂的数据库架构。上传你的课程资料，我可以提供更准确的辅导。</p>
        </div>

        <div 
          v-for="(msg, idx) in messages" 
          :key="idx"
          :class="['flex', msg.role === 'user' ? 'justify-end' : 'justify-start']"
        >
          <div :class="['flex max-w-[80%] space-x-4', msg.role === 'user' ? 'flex-row-reverse space-x-reverse' : 'flex-row']">
            <div :class="['w-10 h-10 rounded-xl flex items-center justify-center flex-shrink-0', msg.role === 'user' ? 'bg-slate-900 text-white' : 'bg-indigo-600 text-white']">
              <User v-if="msg.role === 'user'" class="w-5 h-5" />
              <Bot v-else class="w-5 h-5" />
            </div>
            <div class="space-y-2 min-w-0">
              <div 
                :class="[
                  'p-4 rounded-2xl text-sm leading-relaxed min-w-0',
                  msg.role === 'user' ? 'bg-white border border-slate-200 text-slate-900 shadow-sm whitespace-pre-wrap' : 'bg-white border border-slate-200 text-slate-900 shadow-sm'
                ]"
              >
                <MarkdownMessage v-if="msg.role === 'assistant' && !msg.plain" :content="msg.content" />
                <template v-else>{{ msg.content }}</template>
              </div>
              <div v-if="msg.isRAG" class="flex items-center space-x-1 text-[10px] font-bold text-indigo-500 uppercase tracking-tighter">
                <Brain class="w-3 h-3" />
                <span>基于知识库回答</span>
              </div>
            </div>
          </div>
        </div>

        <div v-if="isLoading" class="flex justify-start">
          <div class="flex max-w-[80%] space-x-4">
             <div class="w-10 h-10 rounded-xl bg-indigo-600 text-white flex items-center justify-center">
               <Bot class="w-5 h-5" />
             </div>
             <div class="p-4 rounded-2xl bg-white border border-slate-200 shadow-sm flex items-center space-x-2">
               <div class="w-1.5 h-1.5 bg-slate-300 rounded-full animate-bounce" />
               <div class="w-1.5 h-1.5 bg-slate-300 rounded-full animate-bounce [animation-delay:-0.15s]" />
               <div class="w-1.5 h-1.5 bg-slate-300 rounded-full animate-bounce [animation-delay:-0.3s]" />
             </div>
          </div>
        </div>
      </div>

      <!-- Input Area -->
      <div class="p-8 bg-gradient-to-t from-slate-50 to-transparent">
        <div class="max-w-4xl mx-auto flex items-end space-x-4 bg-white p-2 pl-4 rounded-2xl shadow-lg border border-slate-200">
          <textarea 
            v-model="input"
            @keydown.enter.prevent="handleSend"
            placeholder="输入你的问题，例如：如何编写三层嵌套查询？"
            class="flex-1 max-h-48 min-h-[44px] py-3 text-sm focus:outline-none resize-none bg-transparent"
          />
          <button 
            @click="handleSend"
            :disabled="!input.trim() || isLoading"
            class="p-3 bg-indigo-600 text-white rounded-xl hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-md active:scale-95"
          >
            <Send class="w-5 h-5" />
          </button>
        </div>
        <p class="text-center text-[10px] text-slate-400 mt-4 font-medium uppercase tracking-widest">
          由通义千问（qwen3.6-plus）与 Jina/Pinecone RAG 提供动力
        </p>
      </div>
    </div>
  </div>
</template>

<style>
.hole {
  top: 4px;
}
</style>
