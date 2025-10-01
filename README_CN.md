# 深度伪造检测平台

一个基于FastAPI构建的综合性深度伪造检测平台，集成了多种深度学习模型用于检测被篡改的图像和视频。

## 🚀 主要功能

- **多模型支持**：VGG、LRCN、Swin Transformer、Vision Transformer、ResNet
- **实时检测**：单文件和批量处理能力
- **视频分析**：逐帧视频分析与结果聚合
- **模型训练**：自动化训练管道，支持进度跟踪
- **数据集管理**：上传、处理和管理数据集
- **RESTful API**：完整的API接口
- **数据库集成**：SQLAlchemy ORM与迁移支持
- **结构化日志**：高级日志记录功能
- **后台处理**：训练和推理的异步任务处理

## 📁 项目结构

```
deepfake-srtp/
├── app/
│   ├── api/                    # API路由
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
│       ├── detection_service.py    # 检测服务
│       ├── training_service.py     # 训练服务
│       ├── model_service.py        # 模型服务
│       └── dataset_service.py      # 数据集服务
├── data/                       # 数据目录
├── models/                     # 模型存储
├── uploads/                    # 文件上传
├── logs/                       # 日志文件
├── requirements.txt            # Python依赖
├── .env.example               # 环境变量模板
├── alembic.ini                # 数据库迁移配置
├── run.py                     # 应用启动脚本
└── README.md                  # 英文文档
```

## 🛠️ 安装指南

1. **克隆仓库**
   ```bash
   git clone <repository-url>
   cd deepfake-srtp
   ```

2. **创建虚拟环境**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Unix/macOS
   source venv/bin/activate
   ```

3. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

4. **设置环境变量**
   ```bash
   cp .env.example .env
   # 编辑 .env 文件配置您的设置
   ```

5. **初始化数据库**
   ```bash
   # 创建数据库表
   python -c "from app.core.database import init_db; init_db()"
   
   # 或使用Alembic进行迁移
   alembic upgrade head
   ```

## 🚀 运行应用

### 开发模式
```bash
python run.py
```

### 生产模式
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### 使用Docker
```bash
docker build -t deepfake-detection .
docker run -p 8000:8000 deepfake-detection
```

## 📚 API文档

应用启动后，访问：

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## 🔧 配置说明

主要环境变量：

| 变量名 | 描述 | 默认值 |
|----------|-------------|---------|
| `DATABASE_URL` | 数据库连接字符串 | `sqlite:///./deepfake_detection.db` |
| `DEFAULT_MODEL_TYPE` | 默认检测模型 | `vgg` |
| `MODEL_INPUT_SIZE` | 模型输入图像尺寸 | `224` |
| `MAX_CONCURRENT_TRAINING_JOBS` | 最大并发训练任务数 | `2` |
| `GPU_ENABLED` | 启用GPU加速 | `True` |

## 🤖 支持的模型

1. **VGG-based CNN**: 用于图像分类的卷积神经网络
2. **LRCN**: 用于视频分析的长时递归卷积网络
3. **Swin Transformer**: 分层视觉变换器
4. **Vision Transformer (ViT)**: 纯变换器架构
5. **ResNet**: 带有跳跃连接的残差网络

## 📊 API端点

### 检测相关
- `POST /api/v1/detection/detect` - 单文件深度伪造检测
- `POST /api/v1/detection/detect-batch` - 批量检测
- `POST /api/v1/detection/detect-video` - 视频检测
- `GET /api/v1/detection/history` - 检测历史记录
- `GET /api/v1/detection/statistics` - 检测统计信息

### 训练相关
- `POST /api/v1/training/jobs` - 创建训练任务
- `GET /api/v1/training/jobs` - 获取训练任务列表
- `GET /api/v1/training/jobs/{id}` - 获取特定训练任务
- `GET /api/v1/training/jobs/{id}/progress` - 训练进度
- `GET /api/v1/training/metrics` - 训练指标

### 模型管理
- `GET /api/v1/models/` - 获取模型列表
- `POST /api/v1/models/` - 创建新模型
- `GET /api/v1/models/{id}` - 获取特定模型
- `POST /api/v1/models/{id}/deploy` - 部署模型
- `GET /api/v1/models/statistics` - 模型统计信息

### 数据集管理
- `GET /api/v1/datasets/` - 获取数据集列表
- `POST /api/v1/datasets/upload` - 上传数据集
- `GET /api/v1/datasets/{id}` - 获取特定数据集
- `POST /api/v1/datasets/{id}/process` - 处理数据集

## 🧪 测试

```bash
# 运行所有测试
pytest

# 运行带覆盖率的测试
pytest --cov=app

# 运行特定测试文件
pytest tests/test_detection.py
```

## 📝 开发指南

### 代码风格
```bash
# 格式化代码
black app/
isort app/

# 代码检查
flake8 app/

# 类型检查
mypy app/
```

### 数据库迁移
```bash
# 创建新迁移
alembic revision --autogenerate -m "描述"

# 应用迁移
alembic upgrade head

# 回滚迁移
alembic downgrade -1
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

result = response.json()
print(f"检测结果: {result['is_fake']}")
print(f"置信度: {result['confidence']}")
```

### 2. 创建训练任务
```python
import requests

training_data = {
    "name": "我的训练任务",
    "model_type": "vgg",
    "dataset_id": "dataset_uuid",
    "config": {
        "epochs": 50,
        "batch_size": 32,
        "learning_rate": 0.001
    }
}

response = requests.post(
    'http://localhost:8000/api/v1/training/jobs',
    json=training_data
)

job = response.json()
print(f"训练任务ID: {job['id']}")
```

### 3. 视频检测
```python
import requests

with open('test_video.mp4', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/v1/detection/detect-video',
        files={'file': f},
        data={'model_type': 'lrcn'}
    )

result = response.json()
print(f"视频检测结果: {result['overall_result']}")
print(f"处理帧数: {result['frames_processed']}")
```

## 🔍 技术架构

### 分层架构
- **API层**: 处理HTTP请求和响应
- **服务层**: 实现业务逻辑
- **模型层**: 数据模型和机器学习模型
- **核心层**: 配置、日志、数据库等基础设施

### 设计模式
- **依赖注入**: 使用FastAPI的依赖注入系统
- **工厂模式**: 模型创建和配置
- **观察者模式**: 训练进度监控
- **策略模式**: 不同检测算法的实现

## 🚀 性能优化

### GPU加速
- 自动检测GPU可用性
- 支持CUDA加速推理和训练
- 批处理优化

### 异步处理
- 异步文件上传和处理
- 后台任务队列
- 并发请求处理

### 缓存策略
- 模型缓存
- 结果缓存
- 数据库查询优化

## 🛡️ 安全特性

- 文件类型验证
- 文件大小限制
- API访问控制
- 输入数据验证
- 错误信息脱敏

## 🤝 贡献指南

1. Fork 本仓库
2. 创建功能分支 (`git checkout -b feature/新功能`)
3. 提交更改 (`git commit -m '添加新功能'`)
4. 推送到分支 (`git push origin feature/新功能`)
5. 创建 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- 原始深度伪造检测研究和模型
- FastAPI框架作为API骨干
- PyTorch和TensorFlow的深度学习实现
- OpenCV的图像和视频处理

## 📞 技术支持

如需支持和帮助：
- 在GitHub仓库中创建issue
- 查看 `/docs` 下的API文档
- 查看 `.env.example` 中的配置选项

## 🔮 未来规划

- [ ] 支持更多深度学习模型
- [ ] 实时视频流检测
- [ ] Web界面管理后台
- [ ] 分布式训练支持
- [ ] 模型性能基准测试
- [ ] 多语言支持

---

**注意**: 本平台仅用于研究和教育目的。请始终验证结果并负责任地使用。
