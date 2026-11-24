const { app, BrowserWindow, dialog, session } = require('electron');
const http = require('http');
const fs = require('fs');
const path = require('path');

const MIME_TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'application/javascript; charset=utf-8',
  '.mjs': 'application/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.ico': 'image/x-icon',
  '.wasm': 'application/wasm',
  '.task': 'application/octet-stream',
  '.tflite': 'application/octet-stream',
};

let localServer = null;
let mainWindow = null;

function getExecutableDir() {
  // electron-builder sets PORTABLE_EXECUTABLE_DIR for portable .exe launches
  if (process.env.PORTABLE_EXECUTABLE_DIR) {
    return process.env.PORTABLE_EXECUTABLE_DIR;
  }
  return path.dirname(app.getPath('exe'));
}

function resolveModelOverride(requestPath) {
  const prefix = '/mediapipe/models/';
  if (!requestPath.startsWith(prefix)) return null;

  const modelName = path.basename(requestPath.slice(prefix.length));
  if (!modelName) return null;

  const exeDir = getExecutableDir();
  const candidates = [
    path.join(exeDir, 'Models', modelName),
    path.join(exeDir, 'models', modelName),
    path.join(exeDir, modelName),
  ];

  for (const candidate of candidates) {
    if (fs.existsSync(candidate) && fs.statSync(candidate).isFile()) {
      return candidate;
    }
  }
  return null;
}

function startStaticServer(distDir) {
  return new Promise((resolve, reject) => {
    const resolvedDist = path.resolve(distDir);

    const server = http.createServer((req, res) => {
      try {
        const parsedUrl = new URL(req.url || '/', 'http://127.0.0.1');
        let pathname = decodeURIComponent(parsedUrl.pathname);
        if (pathname === '/') pathname = '/index.html';

        // 1. Check external Models folder next to the portable .exe first
        const externalModelPath = resolveModelOverride(pathname);
        let filePath = externalModelPath;

        // 2. Otherwise serve bundled assets from dist/
        if (!filePath) {
          const safePath = path.resolve(resolvedDist, `.${pathname}`);
          if (!safePath.startsWith(resolvedDist)) {
            res.writeHead(403);
            res.end('Forbidden');
            return;
          }
          filePath = safePath;
        }

        if (!fs.existsSync(filePath) || fs.statSync(filePath).isDirectory()) {
          filePath = path.join(resolvedDist, 'index.html');
        }

        const ext = path.extname(filePath).toLowerCase();
        const contentType = MIME_TYPES[ext] || 'application/octet-stream';
        res.writeHead(200, {
          'Content-Type': contentType,
          'Cache-Control': 'no-cache',
        });
        fs.createReadStream(filePath).pipe(res);
      } catch {
        res.writeHead(500);
        res.end('Internal server error');
      }
    });

    server.on('error', reject);
    server.listen(0, '127.0.0.1', () => {
      const address = server.address();
      if (!address || typeof address === 'string') {
        reject(new Error('Failed to bind local server port.'));
        return;
      }
      resolve({ server, port: address.port });
    });
  });
}

async function createWindow() {
  const distDir = path.join(__dirname, '..', 'dist');
  const { server, port } = await startStaticServer(distDir);
  localServer = server;

  const unpackedIcon = path.join(
    __dirname.replace('app.asar', 'app.asar.unpacked'),
    '..',
    'dist',
    'app.ico'
  );
  const iconPath = fs.existsSync(unpackedIcon)
    ? unpackedIcon
    : path.join(distDir, 'app.ico');

  mainWindow = new BrowserWindow({
    width: 1120,
    height: 780,
    minWidth: 940,
    minHeight: 680,
    backgroundColor: '#121214',
    title: 'US Visa Photo Automator - Smart Canvas',
    icon: iconPath,
    autoHideMenuBar: true,
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  // Use native Windows Save As dialog when downloading 600 DPI prints
  session.defaultSession.on('will-download', (_event, item) => {
    const defaultName = item.getFilename() || 'VisaReady.jpg';
    const options = {
      title: 'Save 600 DPI Visa Photo',
      defaultPath: path.join(app.getPath('desktop'), defaultName),
      filters: [{ name: 'JPEG Image', extensions: ['jpg', 'jpeg'] }],
    };
    const savePath = mainWindow
      ? dialog.showSaveDialogSync(mainWindow, options)
      : dialog.showSaveDialogSync(options);

    if (savePath) {
      item.setSavePath(savePath);
    } else {
      item.cancel();
    }
  });

  mainWindow.setMenuBarVisibility(false);
  await mainWindow.loadURL(`http://127.0.0.1:${port}/`);

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

app.whenReady().then(() => {
  createWindow().catch((err) => {
    dialog.showErrorBox('Launch Error', String(err));
    app.quit();
  });
});

app.on('window-all-closed', () => {
  if (localServer) {
    localServer.close();
    localServer = null;
  }
  app.quit();
});
