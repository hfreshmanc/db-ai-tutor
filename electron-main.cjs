const { app, BrowserWindow } = require('electron');
const path = require('path');
const { spawn } = require('child_process');

let mainWindow;
let serverProcess;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    title: "DB-AI Tutor 桌面版",
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
    },
    icon: path.join(__dirname, 'public/favicon.ico') // 如果有图标的话
  });

  // 加载本地运行的服务器地址
  mainWindow.loadURL('http://localhost:3000');

  mainWindow.on('closed', function () {
    mainWindow = null;
  });
}

// 启动后端服务器
function startServer() {
  const isDev = !app.isPackaged;
  const serverPath = isDev 
    ? path.join(__dirname, 'server.ts') 
    : path.join(__dirname, 'server.cjs');
  
  const env = { 
    ...process.env, 
    NODE_ENV: isDev ? 'development' : 'production',
    DIST_PATH: path.join(__dirname, 'dist')
  };

  if (isDev) {
    serverProcess = spawn('npx', ['tsx', serverPath], {
      shell: true,
      env
    });
  } else {
    // 生产环境运行编译后的 cjs
    serverProcess = spawn('node', [serverPath], {
      shell: true,
      env
    });
  }

  serverProcess.stdout.on('data', (data) => {
    console.log(`Server: ${data}`);
  });

  serverProcess.stderr.on('data', (data) => {
    console.error(`Server Error: ${data}`);
  });
}

app.on('ready', () => {
  startServer();
  // 等待服务器启动后再创建窗口
  setTimeout(createWindow, 3000); 
});

app.on('window-all-closed', function () {
  if (process.platform !== 'darwin') {
    if (serverProcess) serverProcess.kill();
    app.quit();
  }
});

app.on('activate', function () {
  if (mainWindow === null) createWindow();
});
