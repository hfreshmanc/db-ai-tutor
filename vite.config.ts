import tailwindcss from '@tailwindcss/vite';
import vue from '@vitejs/plugin-vue';
import path from 'path';
import {defineConfig, loadEnv} from 'vite';

export default defineConfig(({mode}) => {
  const env = loadEnv(mode, '.', '');
  const apiProxyMs = Number(env.DEV_API_PROXY_TIMEOUT_MS || 900000);
  const apiProxyTimeout = Number.isFinite(apiProxyMs) && apiProxyMs > 0 ? apiProxyMs : 900000;
  return {
    plugins: [vue(), tailwindcss()],
    resolve: {
      alias: {
        '@': path.resolve(__dirname, '.'),
      },
    },
    server: {
      /** 与后端 3001 分离；避免本机 3000 常被占用导致 Vite 误占 3001 */
      port: 5173,
      strictPort: true,
      proxy: {
        '/api': {
          /** 使用 127.0.0.1，避免 Windows 上 localhost 解析到 ::1 而后端只监听 IPv4 导致 ECONNREFUSED */
          target: (env.DEV_API_PROXY_TARGET || 'http://127.0.0.1:3001').trim(),
          changeOrigin: true,
          secure: false,
          /** 默认 15min：/api/rag/index 会多轮调 Jina，开发代理默认 ~60s 会误杀长请求 */
          timeout: apiProxyTimeout,
          proxyTimeout: apiProxyTimeout,
        }
      },
      // HMR is disabled in AI Studio via DISABLE_HMR env var.
      // Do not modify — file watching is disabled to prevent flickering during agent edits.
      hmr: process.env.DISABLE_HMR !== 'true',
    },
  };
});
