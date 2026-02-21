# 深度伪造检测平台

一个基于FastAPI构建的综合性深度伪造检测平台，集成了多种深度学习模型用于检测被篡改的图像和视频。

## 🚀 主要功能

- **多模型支持**：VGG、LRCN、Swin Transformer、Vision Transformer、ResNet
- **实时检测**：单文件和批量处理能力
- **视频分析**：逐帧视频分析与结果聚合
- **模型训练**：自动化训练管道，支持进度跟踪
- **数据集管理**：上传、处理和管理数据集
- **特征工程管线**：对图像/视频执行真实特征提取并输出结果文件
- **RESTful API**：完整的API接口
- **数据库集成**：SQLAlchemy ORM与Alembic迁移支持
- **结构化日志**：高级日志记录功能
- **后台处理**：使用FastAPI BackgroundTasks执行异步任务
- **GPU加速**：自动检测和使用GPU进行推理和训练
- **健康检查**：系统状态监控端点

## 📁 项目结构

```
deepfake-srtp/
├── app/
│   ├── api/                    # API路由
│   │   ├── __init__.py        # 路由聚合
│   │   └── routes/
│   │       ├── detection.py    # 检测端点
│   │       ├── training.py     # 训练端点
│   │       ├── models.py       # 模型管理端点
│   │       └── datasets.py     # 数据集管理端点
│   ├── core/                   # 核心功能
│   │   ├── config.py          # 配置管理
│   │   ├── database.py        # 数据库设置
│   │   └── logging.py         # 日志配置
│   ├── models/                 # 数据模型
│   │   ├── database_models.py # SQLAlchemy模型
│   │   └── ml_models.py       # 机器学习模型
│   ├── schemas/                # Pydantic模式
│   │   ├── detection.py       # 检测模式
│   │   ├── training.py        # 训练模式
│   │   ├── models.py          # 模型模式
│   │   └── datasets.py        # 数据集模式
│   └── services/               # 业务逻辑
│       ├── base_service.py    # 基础服务类
│       ├── detection_service.py    # 检测服务
│       ├── training_service.py     # 训练服务
│       ├── model_service.py        # 模型服务
│       └── dataset_service.py      # 数据集服务
├── alembic/                    # 数据库迁移
│   ├── versions/              # 迁移版本文件
│   ├── env.py                 # Alembic环境配置
│   └── script.py.mako         # 迁移脚本模板
├── data/                       # 数据目录
├── models/                     # 模型存储
├── uploads/                    # 文件上传
├── logs/                       # 日志文件
├── requirements.txt            # Python依赖
├── .env.example               # 环境变量模板
├── .env                       # 环境变量配置（需要创建）
├── alembic.ini                # 数据库迁移配置
├── run.py                     # 应用启动脚本
├── init_db.py                 # 数据库初始化脚本
├── test_db_connection.py      # 数据库连接测试
└── README.md                  # 英文文档
```

## 🛠️ 安装指南

### 前置要求

- Python 3.8+
- pip 或 conda
- 可选：CUDA支持的GPU（用于加速）

### 1. 克隆仓库
```bash
git clone https://github.com/180cmchn/deepfake-srtp.git
cd deepfake-srtp
```

### 2. 创建虚拟环境
```bash
# 使用venv
python -m venv venv

# Windows
venv\Scripts\activate

# Unix/macOS
source venv/bin/activate

# 或使用conda
conda create -n deepfake-env python=3.8
conda activate deepfake-env
```

### 3. 安装依赖
```bash
pip install -r requirements.txt
```

### 4. 环境配置
```bash
# 复制环境变量模板
cp .env.example .env

# 编辑 .env 文件配置您的设置
# 建议修改以下配置项：
# - SECRET_KEY: 生产环境中必须更改
# - DATABASE_URL: 根据需要配置数据库
# - GPU_ENABLED: 根据硬件配置设置
```

### 5. 数据库初始化
```bash
# 测试数据库连接
python test_db_connection.py

# 初始化数据库表
python init_db.py

# 或使用Alembic进行迁移
alembic upgrade head
```

## 🚀 运行应用

### 开发模式
```bash
# 使用启动脚本（推荐）
python run.py

# 或直接使用uvicorn
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 生产模式
```bash
# 使用多worker
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4

# 或使用启动脚本
python run.py  # 需要在.env中设置RELOAD=False, WORKERS=4
```

### Docker部署
```bash
# 构建镜像
docker build -t deepfake-detection .

# 运行容器
docker run -p 8000:8000 -v $(pwd)/data:/app/data deepfake-detection
```

## 📚 API文档

应用启动后，访问：

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
- **OpenAPI JSON**: `http://localhost:8000/api/v1/openapi.json`

### 健康检查端点
- **应用状态**: `http://localhost:8000/`
- **基础健康检查**: `http://localhost:8000/health`
- **详细健康检查**: `http://localhost:8000/api/v1/health/health`

## 🔧 配置说明

### 主要环境变量

| 变量名 | 描述 | 默认值 | 推荐设置 |
|----------|-------------|---------|---------|
| `APP_NAME` | 应用名称 | Deepfake Detection Platform | - |
| `DEBUG` | 调试模式 | `True` | 生产环境设为`False` |
| `ENVIRONMENT` | 环境类型 | `development` | 生产环境设为`production` |
| `HOST` | 服务器主机 | `0.0.0.0` | - |
| `PORT` | 服务器端口 | `8000` | - |
| `DATABASE_URL` | 数据库连接字符串 | `sqlite:///./deepfake_detection.db` | 生产环境建议使用PostgreSQL/MySQL |
| `SECRET_KEY` | 应用密钥 | 需要更改 | 生产环境必须更改 |
| `DEFAULT_MODEL_TYPE` | 默认检测模型 | `vgg` | - |
| `MODEL_INPUT_SIZE` | 模型输入图像尺寸 | `224` | - |
| `MAX_CONCURRENT_TRAINING_JOBS` | 最大并发训练任务数 | `2` | 根据GPU内存调整 |
| `GPU_ENABLED` | 启用GPU加速 | `True` | 根据硬件配置 |
| `CUDA_VISIBLE_DEVICES` | 可见GPU设备 | `0` | 多GPU环境配置 |
| `LOG_LEVEL` | 日志级别 | `INFO` | 生产环境建议`WARNING` |
| `BACKEND_CORS_ORIGINS` | 允许的跨域源 | `http://localhost:3000,http://localhost:8000` | 根据前端地址配置 |

### 数据库配置

#### SQLite（默认）
```bash
DATABASE_URL=sqlite:///./deepfake_detection.db
```

#### PostgreSQL
```bash
DATABASE_URL=postgresql://username:password@localhost:5432/deepfake_detection
```

#### MySQL
```bash
DATABASE_URL=mysql+pymysql://username:password@localhost:3306/deepfake_detection
```

### Redis配置（可选）
```bash
REDIS_URL=redis://localhost:6379/0
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0
```

说明：当前默认实现基于 FastAPI `BackgroundTasks`，不依赖 Redis/Celery 即可运行。

## 🤖 支持的模型

### 1. VGG-based CNN
- **用途**: 图像分类的卷积神经网络
- **输入尺寸**: 224x224
- **特点**: 经典架构，训练稳定

### 2. LRCN (Long-term Recurrent Convolutional Network)
- **用途**: 视频分析的长时递归卷积网络
- **输入**: 视频帧序列
- **特点**: 结合CNN和LSTM，适合时序数据

### 3. Swin Transformer
- **用途**: 分层视觉变换器
- **输入尺寸**: 224x224
- **特点**: 计算效率高，性能优异

### 4. Vision Transformer (ViT)
- **用途**: 纯变换器架构
- **输入尺寸**: 224x224
- **特点**: 注意力机制，全局建模能力强

### 5. ResNet
- **用途**: 带有跳跃连接的残差网络
- **输入尺寸**: 224x224
- **特点**: 梯度流动好，深度网络训练稳定

## 📊 API端点

### 检测相关 (`/api/v1/detection`)
- `POST /detect` - 单文件深度伪造检测
- `POST /detect/batch` - 批量检测
- `POST /detect/video` - 视频检测
- `GET /history` - 检测历史记录
- `GET /statistics` - 检测统计信息

### 训练相关 (`/api/v1/training`)
- `POST /jobs` - 创建训练任务
- `GET /jobs` - 获取训练任务列表
- `GET /jobs/{id}` - 获取特定训练任务
- `GET /jobs/{id}/progress` - 训练进度
- `GET /metrics` - 训练指标

### 模型管理 (`/api/v1/models`)
- `GET /` - 获取模型列表
- `POST /` - 创建新模型
- `GET /{id}` - 获取特定模型
- `POST /{id}/deploy` - 部署模型
- `GET /statistics/overview` - 模型统计信息

### 数据集管理 (`/api/v1/datasets`)
- `GET /` - 获取数据集列表
- `POST /upload` - 上传数据集
- `GET /{id}` - 获取特定数据集
- `POST /{id}/process` - 处理数据集

数据集处理会将特征工程结果输出到 `data/features/dataset_<id>_features.json`。

处理管线中的标签推断与数据切分规则：
- 标签推断基于文件路径关键字：fake -> `0`，real -> `1`；若路径同时命中或未命中关键字，则标签记为 `null`。
- fake 关键字包含 `fake`、`deepfake`、`manipulated`、`forged`、`tampered`、`class1`；real 关键字包含 `real`、`authentic`、`original`、`genuine`、`pristine`、`class0`。
- 训练/验证/测试样本数由 `validation_split` 与 `test_split` 计算（默认 `0.2` 与 `0.1`），并做边界保护，尽量保证至少保留 1 条训练样本。

## 🧪 测试

### 运行测试
```bash
# 运行所有测试
pytest

# 运行带覆盖率的测试
pytest --cov=app --cov-report=html

# 运行特定测试文件
pytest -k detection -v
```

### 测试配置
```bash
# 安装测试依赖
pip install pytest pytest-asyncio httpx

# 运行性能测试
pytest tests/test_performance.py -v
```

## 📝 开发指南

### 代码风格
```bash
# 格式化代码
black app/ tests/
isort app/ tests/

# 代码检查
flake8 app/ tests/

# 类型检查
mypy app/

# 安全检查
bandit -r app/
```

### 数据库迁移
```bash
# 创建新迁移
alembic revision --autogenerate -m "描述迁移内容"

# 应用迁移
alembic upgrade head

# 回滚迁移
alembic downgrade -1

# 查看迁移历史
alembic history

# 查看当前版本
alembic current
```

### 日志查看
```bash
# 查看应用日志
tail -f logs/app.log

# 查看结构化日志（JSON格式）
cat logs/app.log | jq

# 查看错误日志
grep "ERROR" logs/app.log
```

## 🎯 使用示例

### 1. 检测单张图片
```python
import requests

# 上传图片进行检测
with open('test_image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/v1/detection/detect',
        files={'file': f},
        data={'model_type': 'vgg'}
    )

if response.status_code == 200:
    result = response.json()
    print(f"检测结果: {result['result']['prediction']}")
    print(f"置信度: {result['result']['confidence']:.4f}")
    print(f"处理时间: {result['processing_time']:.2f}s")
else:
    print(f"检测失败: {response.text}")
```

### 2. 创建训练任务
```python
import requests
import json

training_data = {
    "name": "我的训练任务",
    "model_type": "vgg",
    "dataset_path": "D:/datasets/deepfake_train",
    "parameters": {
        "epochs": 50,
        "batch_size": 32,
        "learning_rate": 0.001,
        "validation_split": 0.2
    }
}

response = requests.post(
    'http://localhost:8000/api/v1/training/jobs',
    params={'auto_start': 'true'},
    json=training_data
)

if response.status_code == 200:
    job = response.json()
    print(f"训练任务ID: {job['id']}")
    print(f"状态: {job['status']}")
    
    # 查询训练进度
    progress_response = requests.get(f'http://localhost:8000/api/v1/training/jobs/{job["id"]}/progress')
    if progress_response.status_code == 200:
        progress = progress_response.json()
        print(f"进度: {progress['progress']:.1f}%")
        print(f"当前epoch: {progress['current_epoch']}")
else:
    print(f"创建训练任务失败: {response.text}")
```

### 3. 视频检测
```python
import requests

with open('test_video.mp4', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/v1/detection/detect/video',
        files={'file': f},
        data={
            'model_type': 'lrcn',
            'frame_extraction_interval': '4',
            'max_frames': '20'
        }
    )

if response.status_code == 200:
    result = response.json()
    print(f"视频检测结果: {result['aggregated_result']['prediction']}")
    print(f"处理帧数: {result['summary']['frames_analyzed']}")
    print(f"平均置信度: {result['aggregated_result']['confidence']:.4f}")
    print(f"处理时间: {result['processing_time']:.2f}s")
else:
    print(f"视频检测失败: {response.text}")
```

### 4. 批量检测
```python
import requests
import os

# 准备多个文件
files = []
for filename in ['image1.jpg', 'image2.jpg', 'image3.jpg']:
    if os.path.exists(filename):
        files.append(('files', open(filename, 'rb')))

response = requests.post(
    'http://localhost:8000/api/v1/detection/detect/batch',
    files=files,
    data={'model_type': 'vgg'}
)

if response.status_code == 200:
    results = response.json()
    print(f"批量检测完成，处理了 {len(results['results'])} 个文件")
    for result in results['results']:
        if result.get('result'):
            print(f"文件: {result['file_info']['file_name']}, 结果: {result['result']['prediction']}, 置信度: {result['result']['confidence']:.4f}")
        else:
            print(f"文件: {result['file_info']['file_name']}, 检测失败: {result.get('error_message')}")
else:
    print(f"批量检测失败: {response.text}")

# 关闭文件
for _, file_obj in files:
    file_obj.close()
```

## 🔍 技术架构

### 分层架构
- **API层**: 处理HTTP请求和响应，参数验证
- **服务层**: 实现业务逻辑，协调各组件
- **模型层**: 数据模型和机器学习模型
- **核心层**: 配置、日志、数据库等基础设施

### 设计模式
- **依赖注入**: 使用FastAPI的依赖注入系统
- **工厂模式**: 模型创建和配置
- **观察者模式**: 训练进度监控
- **策略模式**: 不同检测算法的实现
- **仓储模式**: 数据访问抽象

### 数据流
```
客户端请求 → API路由 → 服务层 → 模型层 → 数据库/文件系统
                    ↓
                日志记录 ← 响应返回 ← 结果处理 ← 推理/训练
```

## 🚀 性能优化

### GPU加速
- 自动检测CUDA可用性
- 支持在可用GPU上执行推理与训练
- 支持按批处理参数控制显存占用

### 异步处理
- FastAPI异步路由
- 异步文件I/O操作
- 后台任务（BackgroundTasks）

### 缓存策略
- 模型权重缓存
- 检测结果缓存
- 数据库查询优化
- 静态文件缓存

### 负载均衡
- 多worker部署
- 请求分发
- 连接池管理

## 🛡️ 安全特性

- **输入验证**: Pydantic模型验证
- **文件安全**: 类型检查与大小限制
- **访问控制**: `X-User-ID` 请求头（开发态默认用户）
- **SQL注入防护**: ORM参数化查询
- **CORS配置**: 跨域请求控制
- **日志审计**: 结构化日志记录关键操作

## 🔧 监控和运维

### 健康检查
```bash
# 应用健康状态
curl http://localhost:8000/health

# 详细健康状态
curl http://localhost:8000/api/v1/health/health

# 数据库连接状态
python test_db_connection.py
```

### 日志监控
```bash
# 实时查看日志
tail -f logs/app.log

# 错误统计
grep "ERROR" logs/app.log | wc -l

# 性能指标
grep "processing_time" logs/app.log | awk '{print $NF}' | sort -n
```

### 性能监控
- 响应时间监控
- 内存使用监控
- GPU利用率监控
- 数据库查询性能

## 🤝 贡献指南

### 开发流程
1. Fork 本仓库
2. 创建功能分支 (`git checkout -b feature/新功能`)
3. 提交更改 (`git commit -m '添加新功能'`)
4. 推送到分支 (`git push origin feature/新功能`)
5. 创建 Pull Request

### 代码规范
- 遵循PEP 8代码风格
- 编写单元测试
- 添加类型注解
- 更新文档
- 通过所有CI检查

### 提交规范
```
feat: 新功能
fix: 修复bug
docs: 文档更新
style: 代码格式
refactor: 重构
test: 测试相关
chore: 构建过程或辅助工具的变动
```

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- 原始深度伪造检测研究和模型
- FastAPI框架作为API骨干
- PyTorch的深度学习实现
- OpenCV的图像和视频处理
- SQLAlchemy的数据库ORM
- Alembic的数据库迁移工具

## 📞 技术支持

如需支持和帮助：
- 在GitHub仓库中创建issue
- 查看 `/docs` 下的API文档
- 查看 `.env.example` 中的配置选项
- 参考 `requirements.txt` 中的依赖版本

## 🔮 未来规划

### 短期目标
- [ ] 完善Web管理界面
- [ ] 添加更多预训练模型
- [ ] 优化视频处理性能
- [ ] 增强数据集管理功能

### 中期目标
- [ ] 实时视频流检测
- [ ] 分布式训练支持
- [ ] 模型性能基准测试
- [ ] 移动端API优化

### 长期目标
- [ ] 多语言支持
- [ ] 云端部署方案
- [ ] 边缘计算支持
- [ ] 联邦学习框架

---

**注意**: 本平台仅用于研究和教育目的。请始终验证结果并负责任地使用。

**免责声明**: 本项目不对检测结果的法律效力负责，检测结果仅供参考。
