# Ubuntu 空白远程机部署指南

本文档给出 `deepfake-srtp` 后端和 `deepfake-srtp-fronted` 前端在 **Ubuntu 空白远程机** 上的推荐部署步骤，覆盖环境准备、后端与前端配置、数据库初始化、systemd、Nginx、健康检查，以及测试命令。

> 推荐部署形态：后端通过 `systemd` 监听 `127.0.0.1:8000`，前端由 `nginx` 直接托管静态文件，并由 `nginx` 反代 `/api/v1`、`/health`、`/docs` 等接口。

## 1. 适用范围与前提

- 服务器系统：Ubuntu 22.04 / 24.04
- Python：建议 3.9+，优先 3.10（仓库当前 torch 版本要求 Python 3.9+）
- 同机部署后端、前端、Nginx
- 后端默认先使用 SQLite 单机基线方案
- 若后续并发写入、训练任务和历史写入压力较大，可切换到 MySQL

### 重要说明

1. 后端真实启动入口是 `python run.py`
2. 数据库初始化优先使用 `python init_db.py`
3. 前端是纯静态站点，`npm run build` **不会**生成真正的 `dist/` 目录
4. `.env.example` 中的 `TRAINING_DEVICE=cuda` 不适合空白 CPU 机器，首次部署请改成 `auto` 或 `cpu`
5. 后端会暴露 `/uploads` 和 `/models` 静态目录，不建议直接把后端端口裸露到公网

## 2. 推荐部署结构

```text
Browser
  -> Nginx :80 / :443
    -> static frontend (/opt/deepfake/deepfake-srtp-fronted)
    -> proxy /api/v1, /health, /docs, /redoc, /uploads, /models
      -> backend systemd service (127.0.0.1:8000)
        -> /opt/deepfake/deepfake-srtp
        -> sqlite:///./deepfake_detection.db  (baseline)
```

## 3. 安装系统包

```bash
sudo apt-get update

sudo apt-get install -y \
  git curl nginx ffmpeg sqlite3 \
  libgl1 libglib2.0-0 \
  python3 python3-venv python3-pip \
  build-essential
```

如果需要前端本地调试或运行前端静态校验，请额外安装 Node.js：

```bash
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs
```

### 基础检查

```bash
python3 --version
pip3 --version
node -v
npm -v
nginx -v
ffmpeg -version | head -n 1
```

如果是 NVIDIA/CUDA 机器，再额外检查：

```bash
nvidia-smi
```

## 4. 创建部署用户与目录

```bash
sudo useradd -m -d /opt/deepfake -s /bin/bash deepfake || true
sudo mkdir -p /opt/deepfake
sudo chown -R deepfake:deepfake /opt/deepfake
```

## 5. 拉取仓库

```bash
sudo -u deepfake -H bash -lc '
cd /opt/deepfake
git clone https://github.com/180cmchn/deepfake-srtp.git
git clone https://github.com/180cmchn/deepfake-srtp-fronted.git
'
```

检查目录：

```bash
sudo -u deepfake -H bash -lc '
ls -la /opt/deepfake
ls -la /opt/deepfake/deepfake-srtp
ls -la /opt/deepfake/deepfake-srtp-fronted
'
```

## 6. 安装后端依赖

### 6.1 CPU / 普通机器

```bash
sudo -u deepfake -H bash -lc '
cd /opt/deepfake/deepfake-srtp
python3 -m venv venv
. venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
'
```

### 6.2 CUDA 机器

`requirements.txt` 已注明 Blackwell / CUDA 12.8 机器需要先单独安装 torch / torchvision：

```bash
sudo -u deepfake -H bash -lc '
cd /opt/deepfake/deepfake-srtp
python3 -m venv venv
. venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.7.0 torchvision==0.22.0
pip install -r requirements.txt
'
```

### 6.3 后端依赖检查

```bash
sudo -u deepfake -H bash -lc '
cd /opt/deepfake/deepfake-srtp
venv/bin/python - <<PY
import fastapi, sqlalchemy, cv2, torch
print("fastapi ok")
print("sqlalchemy ok")
print("cv2 ok")
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
PY
'
```

## 7. 安装前端依赖

```bash
sudo -u deepfake -H bash -lc '
cd /opt/deepfake/deepfake-srtp-fronted
npm install
'
```

检查前端依赖：

```bash
sudo -u deepfake -H bash -lc '
cd /opt/deepfake/deepfake-srtp-fronted
npm ls --depth=0
'
```

## 8. 配置后端 `.env`

先生成安全的密钥：

```bash
SECRET_KEY="$(python3 -c 'import secrets; print(secrets.token_urlsafe(64))')"
echo "$SECRET_KEY"
```

然后写入 `.env`：

```bash
sudo -u deepfake -H bash -lc "cat > /opt/deepfake/deepfake-srtp/.env <<EOF
APP_NAME=Deepfake Detection Platform
APP_VERSION=1.0.0
DEBUG=False
ENVIRONMENT=production

HOST=127.0.0.1
PORT=8000
WORKERS=1
RELOAD=False

DATABASE_URL=sqlite:///./deepfake_detection.db

SECRET_KEY=$SECRET_KEY
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

API_V1_STR=/api/v1
BACKEND_CORS_ORIGINS=http://localhost,http://127.0.0.1,http://YOUR_SERVER_IP,https://YOUR_DOMAIN

DEFAULT_MODEL_TYPE=vgg
MODEL_INPUT_SIZE=224
MODEL_USE_PRETRAINED_WEIGHTS=True
MODEL_ALLOW_RANDOM_INIT_FALLBACK=True
MAX_CONCURRENT_TRAINING_JOBS=1

MAX_FILE_SIZE=100
UPLOAD_DIR=uploads
DATA_DIR=data
MODEL_DIR=models
MODELS_DIR=models
LOG_DIR=logs

LOG_LEVEL=INFO
LOG_FORMAT=json

GPU_ENABLED=False
CUDA_VISIBLE_DEVICES=
BATCH_SIZE=8
LEARNING_RATE=0.001
EPOCHS=50
VALIDATION_SPLIT=0.2
TRAINING_DEVICE=auto
TRAINING_NUM_WORKERS=2
TRAINING_PREFETCH_FACTOR=2
TRAINING_PERSISTENT_WORKERS=False

ENABLE_METRICS=False
EOF"
```

### 配置注意事项

- CPU 机器：`TRAINING_DEVICE=auto` 或 `TRAINING_DEVICE=cpu`
- CUDA 机器：确认 `nvidia-smi` 和 torch CUDA 可用后再改为 `TRAINING_DEVICE=cuda`
- 如果使用 MySQL，连接串必须写成 `mysql+pymysql://...`，不要写 `mysql://...`

## 9. 配置前端 `config.js`

生产环境推荐前后端同源，通过 Nginx 把 `/api/v1` 反代到后端。这样前端 `config.js` 最简单：

```bash
sudo -u deepfake -H bash -lc "cat > /opt/deepfake/deepfake-srtp-fronted/config.js <<'EOF'
window.__APP_CONFIG__ = Object.assign(
    {
        API_BASE_URL: '/api/v1'
    },
    window.__APP_CONFIG__ || {}
);
EOF"
```

## 10. 初始化数据库

推荐使用 Alembic-first 初始化流程：

```bash
sudo -u deepfake -H bash -lc '
cd /opt/deepfake/deepfake-srtp
venv/bin/python init_db.py
'
```

然后检查连接：

```bash
sudo -u deepfake -H bash -lc '
cd /opt/deepfake/deepfake-srtp
venv/bin/python test_db_connection.py
'
```

如果使用 SQLite，还可以直接检查数据库文件与表：

```bash
sudo -u deepfake -H bash -lc '
cd /opt/deepfake/deepfake-srtp
ls -lh deepfake_detection.db
sqlite3 deepfake_detection.db ".tables"
'
```

## 11. 手动启动后端做 smoke test

```bash
sudo -u deepfake -H bash -lc '
cd /opt/deepfake/deepfake-srtp
. venv/bin/activate
python run.py
'
```

另开一个终端检查：

```bash
curl http://127.0.0.1:8000/health | python3 -m json.tool
curl http://127.0.0.1:8000/api/v1/health/status | python3 -m json.tool
curl -I http://127.0.0.1:8000/docs
curl -I http://127.0.0.1:8000/redoc
curl -I http://127.0.0.1:8000/api/v1/openapi.json
```

确认无误后，停止这个手工进程。

## 12. 配置 systemd 启动后端

创建服务文件：

```bash
sudo tee /etc/systemd/system/deepfake-backend.service > /dev/null <<'EOF'
[Unit]
Description=Deepfake Detection Backend
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=deepfake
Group=deepfake
WorkingDirectory=/opt/deepfake/deepfake-srtp
EnvironmentFile=/opt/deepfake/deepfake-srtp/.env
ExecStart=/opt/deepfake/deepfake-srtp/venv/bin/python run.py
Restart=always
RestartSec=5

NoNewPrivileges=true
PrivateTmp=true

[Install]
WantedBy=multi-user.target
EOF
```

启动并设置开机自启：

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now deepfake-backend
sudo systemctl status deepfake-backend --no-pager
```

查看日志：

```bash
journalctl -u deepfake-backend -f
```

## 13. 配置 Nginx

### 13.1 站点配置

把 `YOUR_DOMAIN` 换成你的域名；如果没有域名，可先用服务器 IP 或 `_`：

```bash
sudo tee /etc/nginx/sites-available/deepfake > /dev/null <<'EOF'
server {
    listen 80;
    server_name YOUR_DOMAIN _;

    root /opt/deepfake/deepfake-srtp-fronted;
    index index.html;

    client_max_body_size 100m;

    location / {
        try_files $uri $uri/ /index.html;
    }

    location /api/v1/ {
        proxy_pass http://127.0.0.1:8000/api/v1/;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location = /health {
        proxy_pass http://127.0.0.1:8000/health;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /docs {
        proxy_pass http://127.0.0.1:8000/docs;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
    }

    location /redoc {
        proxy_pass http://127.0.0.1:8000/redoc;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
    }

    location = /api/v1/openapi.json {
        proxy_pass http://127.0.0.1:8000/api/v1/openapi.json;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
    }

    location /uploads/ {
        proxy_pass http://127.0.0.1:8000/uploads/;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
    }

    location /models/ {
        proxy_pass http://127.0.0.1:8000/models/;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
    }
}
EOF
```

### 13.2 启用站点

```bash
sudo ln -sf /etc/nginx/sites-available/deepfake /etc/nginx/sites-enabled/deepfake
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl reload nginx
sudo systemctl status nginx --no-pager
```

## 14. 最终联通检查

```bash
curl http://YOUR_DOMAIN/health | python3 -m json.tool
curl -I http://YOUR_DOMAIN/
curl -I http://YOUR_DOMAIN/docs
curl -I http://YOUR_DOMAIN/redoc
curl -I http://YOUR_DOMAIN/api/v1/openapi.json
```

浏览器检查：

- `http://YOUR_DOMAIN/`
- `http://YOUR_DOMAIN/docs`
- `http://YOUR_DOMAIN/redoc`

## 15. 测试命令

### 15.1 后端测试

当前仓库最稳妥的内置测试方式是 `unittest discover`：

```bash
sudo -u deepfake -H bash -lc '
cd /opt/deepfake/deepfake-srtp
venv/bin/python -m unittest discover -s tests -p "test_*.py"
'
```

按模块运行也可以：

```bash
sudo -u deepfake -H bash -lc '
cd /opt/deepfake/deepfake-srtp
venv/bin/python -m unittest tests.test_health_truthfulness
venv/bin/python -m unittest tests.test_config_alignment
venv/bin/python -m unittest tests.test_reporting_contract_truthfulness
'
```

语法检查：

```bash
sudo -u deepfake -H bash -lc '
cd /opt/deepfake/deepfake-srtp
venv/bin/python -m py_compile \
  app/main.py \
  app/core/config.py \
  app/core/database.py \
  app/api/routes/detection.py \
  app/api/routes/health.py \
  app/services/detection_service.py \
  app/services/training_service.py \
  run.py
'
```

### 15.2 可选的 pytest 路径

如果你希望按 README 中的 pytest 方式运行，请先安装额外测试依赖：

```bash
sudo -u deepfake -H bash -lc '
cd /opt/deepfake/deepfake-srtp
. venv/bin/activate
pip install pytest pytest-asyncio httpx pytest-cov
pytest
pytest --cov=app
'
```

### 15.3 前端验证

```bash
sudo -u deepfake -H bash -lc '
cd /opt/deepfake/deepfake-srtp-fronted
npm run test:static
node --check config.js
node --check script.js
npm run build
'
```

说明：

- `npm run test:static` 是当前前端最直接的静态契约检查
- `node --check` 用于语法检查
- `npm run build` 只会输出 `Build completed - static files ready`，不会生成打包产物

## 16. 日常运维命令

```bash
sudo systemctl status deepfake-backend --no-pager
sudo systemctl status nginx --no-pager
journalctl -u deepfake-backend -f
sudo tail -f /var/log/nginx/access.log /var/log/nginx/error.log
ss -lntp | grep -E '(:80|:8000)'
```

SQLite 检查：

```bash
sudo -u deepfake -H bash -lc '
cd /opt/deepfake/deepfake-srtp
sqlite3 deepfake_detection.db ".tables"
'
```

## 17. 切换到 MySQL（可选）

当你预计会有多人同时使用、较高并发写入、训练任务与历史记录频繁写入时，建议切换到 MySQL。

### 17.1 安装 MySQL

```bash
sudo apt-get install -y mysql-server
sudo systemctl enable --now mysql
sudo systemctl status mysql --no-pager
```

### 17.2 创建库和用户

```bash
sudo mysql -e "CREATE DATABASE deepfake_detection CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"
sudo mysql -e "CREATE USER 'deepfake'@'localhost' IDENTIFIED BY 'CHANGE_THIS_PASSWORD';"
sudo mysql -e "GRANT ALL PRIVILEGES ON deepfake_detection.* TO 'deepfake'@'localhost';"
sudo mysql -e "FLUSH PRIVILEGES;"
```

### 17.3 修改 `.env`

把：

```env
DATABASE_URL=sqlite:///./deepfake_detection.db
```

改成：

```env
DATABASE_URL=mysql+pymysql://deepfake:CHANGE_THIS_PASSWORD@127.0.0.1:3306/deepfake_detection
```

### 17.4 重跑初始化与连接检查

```bash
sudo -u deepfake -H bash -lc '
cd /opt/deepfake/deepfake-srtp
venv/bin/python init_db.py
venv/bin/python test_db_connection.py
'

sudo systemctl restart deepfake-backend
sudo systemctl status deepfake-backend --no-pager
```

## 18. 常见坑

1. 不要直接按 `.env.example` 原样上线，特别是 `TRAINING_DEVICE=cuda`
2. 当前仓库没有完整应用 Dockerfile，不要直接照抄旧的 `docker build` 指令
3. 不建议直接暴露后端 `8000` 到公网
4. `npm run dev` 适合本地调试，不是生产部署方案
5. 如果目标机器无法访问外网 CDN，前端页面需要本地化 Tailwind / Axios / Chart.js / Font Awesome 等资源
6. 若通过 `systemd` 部署，日志优先看 `journalctl -u deepfake-backend -f`

## 19. 本地前端联动远程后端（可选）

如果后端部署在远程云主机而前端只在本地运行，可以在本地建立 SSH 隧道：

```bash
ssh -p 13114 -L 8000:127.0.0.1:8000 root@connect.westd.seetacloud.com
```

建立后，本地前端仍可以使用 `http://localhost:8000/api/v1` 访问远程后端。
