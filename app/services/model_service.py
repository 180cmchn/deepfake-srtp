"""
Model management service for deepfake detection platform
"""

import os
import time
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session

from app.core.logging import logger
from app.models.database_models import ModelRegistry, TrainingJob
from app.schemas.models import (
    ModelCreate, ModelResponse, ModelList, ModelUpdate,
    ModelStatistics, ModelDeployment, ModelStatus, ModelMetrics
)


class ModelService:
    """Service for managing model registry and deployment"""
    
    def __init__(self, db: Session):
        self.db = db
    
    async def create_model(self, model: ModelCreate) -> ModelResponse:
        """Create a new model registry entry"""
        try:
            # Check if model name already exists
            existing = self.db.query(ModelRegistry).filter(
                ModelRegistry.name == model.name,
                ModelRegistry.del_flag == 0
            ).first()
            
            if existing:
                raise ValueError(f"Model with name '{model.name}' already exists")
            
            # Create model registry entry
            db_model = ModelRegistry(
                name=model.name,
                model_type=model.model_type,
                version=model.version,
                file_path=model.file_path,
                description=model.description,
                input_size=model.input_size,
                num_classes=model.num_classes,
                parameters=model.parameters,
                training_job_id=model.training_job_id
            )
            
            self.db.add(db_model)
            self.db.commit()
            self.db.refresh(db_model)
            
            logger.info("Model created", model_name=model.name, model_type=model.model_type)
            
            return self._db_to_response(db_model)
            
        except Exception as e:
            logger.error("Failed to create model", error=str(e))
            self.db.rollback()
            raise
    
    async def get_models(
        self,
        skip: int = 0,
        limit: int = 100,
        model_type: Optional[str] = None,
        status: Optional[str] = None
    ) -> ModelList:
        """Get models list"""
        try:
            query = self.db.query(ModelRegistry).filter(ModelRegistry.del_flag == 0)
            
            if model_type:
                query = query.filter(ModelRegistry.model_type == model_type)
            
            if status:
                query = query.filter(ModelRegistry.status == status)
            
            total = query.count()
            results = query.order_by(ModelRegistry.created_at.desc()).offset(skip).limit(limit).all()
            
            models = [self._db_to_response(model) for model in results]
            
            return ModelList(
                models=models,
                total=total,
                page=skip // limit + 1,
                size=limit,
                pages=(total + limit - 1) // limit
            )
            
        except Exception as e:
            logger.error("Failed to get models", error=str(e))
            raise
    
    async def get_model(self, model_id: int) -> Optional[ModelResponse]:
        """Get model by ID"""
        try:
            model = self.db.query(ModelRegistry).filter(
                ModelRegistry.id == model_id,
                ModelRegistry.del_flag == 0
            ).first()
            
            if not model:
                return None
            
            return self._db_to_response(model)
            
        except Exception as e:
            logger.error("Failed to get model", error=str(e), model_id=model_id)
            raise
    
    async def update_model(self, model_id: int, model_update: ModelUpdate) -> Optional[ModelResponse]:
        """Update model"""
        try:
            model = self.db.query(ModelRegistry).filter(
                ModelRegistry.id == model_id,
                ModelRegistry.del_flag == 0
            ).first()
            
            if not model:
                return None
            
            # Update fields
            update_data = model_update.dict(exclude_unset=True)
            for field, value in update_data.items():
                if hasattr(model, field):
                    setattr(model, field, value)
            
            self.db.commit()
            self.db.refresh(model)
            
            logger.info("Model updated", model_id=model_id)
            
            return self._db_to_response(model)
            
        except Exception as e:
            logger.error("Failed to update model", error=str(e), model_id=model_id)
            self.db.rollback()
            raise
    
    async def delete_model(self, model_id: int) -> bool:
        """Delete model"""
        try:
            model = self.db.query(ModelRegistry).filter(
                ModelRegistry.id == model_id,
                ModelRegistry.del_flag == 0
            ).first()
            
            if not model:
                return False
            
            # Check if model is deployed
            if model.status == ModelStatus.DEPLOYED.value:
                raise ValueError("Cannot delete deployed model. Undeploy first.")
            
            model.del_flag = 1
            self.db.commit()
            
            logger.info("Model deleted", model_id=model_id)
            return True
            
        except Exception as e:
            logger.error("Failed to delete model", error=str(e), model_id=model_id)
            self.db.rollback()
            raise
    
    async def deploy_model(self, model_id: int, deployment_config: Dict[str, Any]) -> ModelDeployment:
        """Deploy model"""
        try:
            model = self.db.query(ModelRegistry).filter(
                ModelRegistry.id == model_id,
                ModelRegistry.del_flag == 0
            ).first()
            
            if not model:
                raise ValueError("Model not found")
            
            if model.status != ModelStatus.READY.value:
                raise ValueError("Only ready models can be deployed")
            
            # Update model status
            model.status = ModelStatus.DEPLOYED.value
            model.deployment_info = deployment_config
            model.is_default = deployment_config.get("is_default", False)
            
            # If this is set as default, unset other defaults
            if model.is_default:
                self.db.query(ModelRegistry).filter(
                    ModelRegistry.id != model_id,
                    ModelRegistry.is_default == True,
                    ModelRegistry.del_flag == 0
                ).update({"is_default": False})
            
            self.db.commit()
            
            deployment = ModelDeployment(
                model_id=model_id,
                deployment_config=deployment_config,
                endpoint_url=f"/api/v1/models/{model_id}/predict",
                health_check_url=f"/api/v1/models/{model_id}/health",
                deployment_status="deployed",
                deployed_at=time.time()
            )
            
            logger.info("Model deployed", model_id=model_id)
            
            return deployment
            
        except Exception as e:
            logger.error("Failed to deploy model", error=str(e), model_id=model_id)
            self.db.rollback()
            raise
    
    async def get_statistics(self) -> ModelStatistics:
        """Get model statistics"""
        try:
            # Get total models
            total_models = self.db.query(ModelRegistry).filter(ModelRegistry.del_flag == 0).count()
            
            # Models by type
            models_by_type = {}
            for model_type in ["vgg", "lrcn", "swin", "vit", "resnet"]:
                count = self.db.query(ModelRegistry).filter(
                    ModelRegistry.model_type == model_type,
                    ModelRegistry.del_flag == 0
                ).count()
                if count > 0:
                    models_by_type[model_type] = count
            
            # Models by status
            models_by_status = {}
            for status in [ModelStatus.TRAINING.value, ModelStatus.READY.value, 
                          ModelStatus.DEPLOYED.value, ModelStatus.ARCHIVED.value, ModelStatus.FAILED.value]:
                count = self.db.query(ModelRegistry).filter(
                    ModelRegistry.status == status,
                    ModelRegistry.del_flag == 0
                ).count()
                if count > 0:
                    models_by_status[status] = count
            
            # Calculate average accuracy
            models_with_accuracy = self.db.query(ModelRegistry.accuracy).filter(
                ModelRegistry.accuracy.isnot(None),
                ModelRegistry.del_flag == 0
            ).all()
            average_accuracy = sum(acc[0] for acc in models_with_accuracy) / len(models_with_accuracy) if models_with_accuracy else None
            
            # Get best model
            best_model = self.db.query(ModelRegistry).filter(
                ModelRegistry.accuracy.isnot(None),
                ModelRegistry.del_flag == 0
            ).order_by(ModelRegistry.accuracy.desc()).first()
            
            # Get recent deployments
            recent_deployments = self.db.query(ModelRegistry).filter(
                ModelRegistry.status == ModelStatus.DEPLOYED.value,
                ModelRegistry.del_flag == 0
            ).order_by(ModelRegistry.updated_at.desc()).limit(5).all()
            
            return ModelStatistics(
                total_models=total_models,
                models_by_type=models_by_type,
                models_by_status=models_by_status,
                average_accuracy=average_accuracy,
                best_model=self._db_to_response(best_model) if best_model else None,
                recent_deployments=[self._db_to_response(m) for m in recent_deployments]
            )
            
        except Exception as e:
            logger.error("Failed to get model statistics", error=str(e))
            raise
    
    def _db_to_response(self, model: ModelRegistry) -> ModelResponse:
        """Convert database model to response schema"""
        metrics = None
        if model.accuracy is not None:
            metrics = ModelMetrics(
                accuracy=model.accuracy,
                precision=model.precision,
                recall=model.recall,
                f1_score=model.f1_score
            )
        
        return ModelResponse(
            id=model.id,
            name=model.name,
            model_type=model.model_type,
            version=model.version,
            description=model.description,
            input_size=model.input_size,
            num_classes=model.num_classes,
            parameters=model.parameters,
            file_path=model.file_path,
            status=ModelStatus(model.status),
            metrics=metrics,
            is_default=model.is_default,
            deployment_info=model.deployment_info,
            created_at=model.created_at,
            updated_at=model.updated_at,
            training_job_id=model.training_job_id
        )
