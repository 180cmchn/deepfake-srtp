# Deepfake Detection Platform

A comprehensive deepfake detection platform built with FastAPI, featuring multiple deep learning models for detecting manipulated images and videos.

## 🚀 Features

- **Multiple Model Support**: VGG, LRCN, Swin Transformer, Vision Transformer, ResNet
- **Real-time Detection**: Single file and batch processing capabilities
- **Video Analysis**: Frame-by-frame video analysis with aggregation
- **Model Training**: Automated training pipeline with progress tracking
- **Dataset Management**: Upload, process, and manage datasets
- **Feature Engineering Pipeline**: Real image/video feature extraction with artifact output
- **RESTful API**: Complete API with comprehensive endpoints
- **Database Integration**: SQLAlchemy ORM with migration support
- **Structured Logging**: Advanced logging with structured output
- **Background Processing**: Async task processing for training and inference

## 📁 Project Structure

```
deepfake-srtp/
├── app/
│   ├── api/                    # API routes
│   │   └── routes/
│   │       ├── detection.py    # Detection endpoints
│   │       ├── training.py     # Training endpoints
│   │       ├── models.py       # Model management endpoints
│   │       └── datasets.py     # Dataset management endpoints
│   ├── core/                   # Core functionality
│   │   ├── config.py          # Configuration management
│   │   ├── database.py        # Database setup
│   │   └── logging.py         # Logging configuration
│   ├── models/                 # Data models
│   │   ├── database_models.py # SQLAlchemy models
│   │   └── ml_models.py       # Machine learning models
│   ├── schemas/                # Pydantic schemas
│   │   ├── detection.py       # Detection schemas
│   │   ├── training.py        # Training schemas
│   │   ├── models.py          # Model schemas
│   │   └── datasets.py        # Dataset schemas
│   └── services/               # Business logic
│       ├── detection_service.py
│       ├── training_service.py
│       ├── model_service.py
│       └── dataset_service.py
├── data/                       # Data directory
├── models/                     # Model storage
├── uploads/                    # File uploads
├── logs/                       # Log files
├── requirements.txt            # Python dependencies
├── .env.example               # Environment variables template
├── alembic.ini                # Database migration config
├── run.py                     # Application startup script
└── README.md                  # This file
```

## 💻 System Requirements

### Minimum Requirements
- **Python**: 3.8 or higher
- **RAM**: 8GB (16GB recommended for training)
- **Storage**: 10GB free space (more for datasets and models)
- **GPU**: Optional, but recommended for training (NVIDIA CUDA-compatible)

### Recommended Requirements
- **Python**: 3.9 or 3.10
- **RAM**: 16GB or more
- **Storage**: 50GB+ SSD
- **GPU**: NVIDIA GPU with 8GB+ VRAM (RTX 3060 or better)

### Supported Operating Systems
- Windows 10/11
- macOS 10.15+
- Ubuntu 18.04+ / CentOS 7+

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/180cmchn/deepfake-srtp.git
   cd deepfake-srtp
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Unix/macOS
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Initialize database**
   ```bash
   # Create database tables
   python -c "from app.core.database import create_tables; create_tables()"
   
   # Or use Alembic for migrations
   alembic upgrade head
   ```

6. **Test database connection**
   ```bash
   python test_db_connection.py
   ```

## 🔧 Troubleshooting

### Common Issues

1. **CUDA/GPU Issues**
   ```bash
   # Check CUDA availability
   python -c "import torch; print(torch.cuda.is_available())"
   
   # If False, install CUDA toolkit or set GPU_ENABLED=False
   ```

2. **Database Connection Issues**
   ```bash
   # Test database connection
   python test_db_connection.py
   
   # For MySQL, ensure pymysql is installed
   pip install pymysql
   ```

3. **Memory Issues**
   - Reduce `BATCH_SIZE` in `.env`
   - Use smaller model input size
   - Close other applications

4. **Import Errors**
   ```bash
   # Reinstall dependencies
   pip install -r requirements.txt --force-reinstall
   ```

### Performance Optimization

1. **Enable GPU acceleration**
   - Install CUDA toolkit
   - Set `GPU_ENABLED=True` in `.env`
   - Use appropriate batch sizes

2. **Database optimization**
   - Use PostgreSQL/MySQL for production
   - Configure connection pooling
   - Enable query logging for debugging

3. **Model optimization**
   - Use model quantization for inference
   - Implement model caching
   - Use appropriate precision (FP16/FP32)

## 🚀 Running the Application

### Development Mode
```bash
python run.py
```

### Production Mode
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Using Docker
```bash
docker build -t deepfake-detection .
docker run -p 8000:8000 deepfake-detection
```

## 📚 API Documentation

Once the application is running, visit:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## 🔧 Configuration

Key environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | Database connection string | `sqlite:///./deepfake_detection.db` |
| `DEFAULT_MODEL_TYPE` | Default model for detection | `vgg` |
| `MODEL_INPUT_SIZE` | Input image size for models | `224` |
| `MAX_CONCURRENT_TRAINING_JOBS` | Max concurrent training jobs | `2` |
| `GPU_ENABLED` | Enable GPU acceleration | `True` |
| `HOST` | Server host address | `0.0.0.0` |
| `PORT` | Server port | `8000` |
| `DEBUG` | Enable debug mode | `False` |
| `LOG_LEVEL` | Logging level | `INFO` |

### Database Configuration

The platform supports multiple database backends:

- **SQLite** (default): `sqlite:///./deepfake_detection.db`
- **PostgreSQL**: `postgresql://user:password@localhost/dbname`
- **MySQL**: `mysql+pymysql://user:password@localhost/dbname`

### GPU Support

For GPU acceleration, ensure CUDA is available and set `GPU_ENABLED=True` in your `.env` file.

## 🤖 Supported Models

1. **VGG-based CNN**: Convolutional Neural Network for image classification
2. **LRCN**: Long-term Recurrent Convolutional Network for video analysis
3. **Swin Transformer**: Hierarchical vision transformer
4. **Vision Transformer (ViT)**: Pure transformer architecture
5. **ResNet**: Residual Network with skip connections

## 📊 API Endpoints

### Detection
- `POST /api/v1/detection/detect` - Detect deepfake in single file
- `POST /api/v1/detection/detect/batch` - Batch detection
- `POST /api/v1/detection/detect/video` - Video detection
- `GET /api/v1/detection/history` - Detection history
- `GET /api/v1/detection/statistics` - Detection statistics

### Training
- `POST /api/v1/training/jobs` - Create training job
- `GET /api/v1/training/jobs` - List training jobs
- `GET /api/v1/training/jobs/{id}` - Get training job
- `GET /api/v1/training/jobs/{id}/progress` - Training progress
- `POST /api/v1/training/jobs/{id}/start` - Start a training job
- `POST /api/v1/training/jobs/{id}/stop` - Stop a training job
- `GET /api/v1/training/jobs/{id}/logs` - Training logs
- `GET /api/v1/training/metrics` - Training metrics
- `GET /api/v1/training/statistics` - Training job statistics

### Models
- `GET /api/v1/models/` - List models
- `POST /api/v1/models/` - Create model
- `GET /api/v1/models/{id}` - Get model
- `POST /api/v1/models/{id}/deploy` - Deploy model
- `GET /api/v1/models/statistics/overview` - Model statistics

### Datasets
- `GET /api/v1/datasets/` - List datasets
- `POST /api/v1/datasets/upload` - Upload dataset
- `GET /api/v1/datasets/{id}` - Get dataset
- `POST /api/v1/datasets/{id}/process` - Process dataset

Dataset processing writes engineered feature artifacts to `data/features/dataset_<id>_features.json`.

Label inference and split rules used by processing pipeline:
- Label inference is path-keyword based: fake -> `0`, real -> `1`; unknown/mixed paths are kept with `null` label.
- Supported image keywords include `fake`, `deepfake`, `manipulated`, `forged`, `tampered`, `class1`; real keywords include `real`, `authentic`, `original`, `genuine`, `pristine`, `class0`.
- Train/validation/test counts are computed from `validation_split` and `test_split` (defaults: `0.2` and `0.1`) with bounds checking to always keep at least one training sample when possible.

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app
```

## 📝 Development

### Code Style
```bash
# Format code
black app/
isort app/

# Lint code
flake8 app/

# Type checking
mypy app/
```

### Database Migrations
```bash
# Create new migration
alembic revision --autogenerate -m "Description"

# Apply migrations
alembic upgrade head

# Rollback migration
alembic downgrade -1
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Original deepfake detection research and models
- FastAPI framework for the API backbone
- PyTorch for deep learning implementations
- OpenCV for image and video processing

## 📞 Support

For support and questions:
- Create an issue in the GitHub repository
- Check the API documentation at `/docs`
- Review the configuration options in `.env.example`

---

**Note**: This platform is for research and educational purposes. Always verify results and use responsibly.
