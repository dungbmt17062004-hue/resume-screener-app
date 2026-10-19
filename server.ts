// Gateway: cổng 3000, chuyển /api/* sang FastAPI (8000), phục vụ frontend Vite.
import express from 'express';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { createProxyMiddleware } from 'http-proxy-middleware';
import { createServer as createVite } from 'vite';

const PORT = 3000, PY_PORT = 8000;
const isProd = process.env.NODE_ENV === 'production';

const py = spawn(process.env.PYTHON ?? 'python3', ['-m', 'uvicorn', 'main:app', '--port', String(PY_PORT)], { stdio: 'inherit' });
const stop = () => { py.kill(); process.exit(0); };
process.on('SIGINT', stop); process.on('SIGTERM', stop);

const app = express();
app.use(createProxyMiddleware({ target: `http://127.0.0.1:${PY_PORT}`, pathFilter: '/api', changeOrigin: true,
  on: { error: (_e, _r, res: any) => res.writeHead?.(503, { 'Content-Type': 'application/json' }).end('{"detail":"Dịch vụ Python chưa sẵn sàng, thử lại sau vài giây."}') } }));

if (isProd) {
  app.use(express.static('dist'));
  app.get('*', (_q, r) => r.sendFile(path.resolve('dist/index.html')));
} else {
  const vite = await createVite({ server: { middlewareMode: true }, appType: 'spa' });
  app.use(vite.middlewares);
}
app.listen(PORT, '0.0.0.0', () => console.log(`Gateway: http://localhost:${PORT}`));
