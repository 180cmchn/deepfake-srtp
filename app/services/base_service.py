"""
Base Service Class
Provides common database operations and utilities
"""

from typing import TypeVar, Generic, Type, List, Optional, Any, Dict
from sqlalchemy.orm import Session
from sqlalchemy import desc, asc, and_, or_
from pydantic import BaseModel
import structlog

logger = structlog.get_logger()

# Type variables for generic service
ModelType = TypeVar("ModelType")
CreateSchemaType = TypeVar("CreateSchemaType", bound=BaseModel)
UpdateSchemaType = TypeVar("UpdateSchemaType", bound=BaseModel)


class BaseService(Generic[ModelType, CreateSchemaType, UpdateSchemaType]):
    """
    Base service class with common CRUD operations
    """
    
    def __init__(self, model: Type[ModelType], db: Session):
        """
        Initialize base service
        
        Args:
            model: SQLAlchemy model class
            db: Database session
        """
        self.model = model
        self.db = db
    
    def get_by_id(self, id: Any) -> Optional[ModelType]:
        """
        Get record by ID
        
        Args:
            id: Record ID
            
        Returns:
            Model instance or None if not found
        """
        try:
            return self.db.query(self.model).filter(self.model.id == id).first()
        except Exception as e:
            logger.error("Failed to get record by ID", 
                        model=self.model.__name__, 
                        id=id, 
                        error=str(e))
            raise
    
    def get_multi(
        self,
        skip: int = 0,
        limit: int = 100,
        order_by: str = "id",
        order_desc: bool = False,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[ModelType]:
        """
        Get multiple records with pagination and filtering
        
        Args:
            skip: Number of records to skip
            limit: Maximum number of records to return
            order_by: Field to order by
            order_desc: Whether to order in descending order
            filters: Dictionary of field filters
            
        Returns:
            List of model instances
        """
        try:
            query = self.db.query(self.model)
            
            # Apply filters
            if filters:
                for field, value in filters.items():
                    if hasattr(self.model, field) and value is not None:
                        query = query.filter(getattr(self.model, field) == value)
            
            # Apply ordering
            if hasattr(self.model, order_by):
                order_column = getattr(self.model, order_by)
                if order_desc:
                    query = query.order_by(desc(order_column))
                else:
                    query = query.order_by(asc(order_column))
            
            # Apply pagination
            return query.offset(skip).limit(limit).all()
            
        except Exception as e:
            logger.error("Failed to get multiple records", 
                        model=self.model.__name__, 
                        error=str(e))
            raise
    
    def create(self, obj_in: CreateSchemaType, **kwargs) -> ModelType:
        """
        Create new record
        
        Args:
            obj_in: Creation schema
            **kwargs: Additional fields to set
            
        Returns:
            Created model instance
            
        Raises:
            Exception: If creation fails
        """
        try:
            obj_data = obj_in.dict() if hasattr(obj_in, 'dict') else obj_in
            obj_data.update(kwargs)
            
            db_obj = self.model(**obj_data)
            self.db.add(db_obj)
            self.db.commit()
            self.db.refresh(db_obj)
            
            logger.info("Record created", 
                       model=self.model.__name__, 
                       id=getattr(db_obj, 'id', None))
            
            return db_obj
            
        except Exception as e:
            self.db.rollback()
            logger.error("Failed to create record", 
                        model=self.model.__name__, 
                        error=str(e))
            raise
    
    def update(
        self,
        db_obj: ModelType,
        obj_in: UpdateSchemaType,
        **kwargs
    ) -> ModelType:
        """
        Update existing record
        
        Args:
            db_obj: Existing model instance
            obj_in: Update schema
            **kwargs: Additional fields to set
            
        Returns:
            Updated model instance
            
        Raises:
            Exception: If update fails
        """
        try:
            obj_data = obj_in.dict(exclude_unset=True) if hasattr(obj_in, 'dict') else obj_in
            obj_data.update(kwargs)
            
            for field, value in obj_data.items():
                if hasattr(db_obj, field):
                    setattr(db_obj, field, value)
            
            self.db.commit()
            self.db.refresh(db_obj)
            
            logger.info("Record updated", 
                       model=self.model.__name__, 
                       id=getattr(db_obj, 'id', None))
            
            return db_obj
            
        except Exception as e:
            self.db.rollback()
            logger.error("Failed to update record", 
                        model=self.model.__name__, 
                        id=getattr(db_obj, 'id', None),
                        error=str(e))
            raise
    
    def delete(self, id: Any) -> bool:
        """
        Delete record by ID
        
        Args:
            id: Record ID to delete
            
        Returns:
            True if deleted successfully, False if not found
            
        Raises:
            Exception: If deletion fails
        """
        try:
            obj = self.get_by_id(id)
            if not obj:
                return False
            
            self.db.delete(obj)
            self.db.commit()
            
            logger.info("Record deleted", 
                       model=self.model.__name__, 
                       id=id)
            
            return True
            
        except Exception as e:
            self.db.rollback()
            logger.error("Failed to delete record", 
                        model=self.model.__name__, 
                        id=id,
                        error=str(e))
            raise
    
    def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """
        Count records with optional filtering
        
        Args:
            filters: Dictionary of field filters
            
        Returns:
            Number of records matching filters
        """
        try:
            query = self.db.query(self.model)
            
            # Apply filters
            if filters:
                for field, value in filters.items():
                    if hasattr(self.model, field) and value is not None:
                        query = query.filter(getattr(self.model, field) == value)
            
            return query.count()
            
        except Exception as e:
            logger.error("Failed to count records", 
                        model=self.model.__name__, 
                        error=str(e))
            raise
    
    def exists(self, id: Any) -> bool:
        """
        Check if record exists by ID
        
        Args:
            id: Record ID to check
            
        Returns:
            True if record exists, False otherwise
        """
        try:
            return self.db.query(self.model).filter(self.model.id == id).first() is not None
        except Exception as e:
            logger.error("Failed to check record existence", 
                        model=self.model.__name__, 
                        id=id,
                        error=str(e))
            raise


class SoftDeleteMixin:
    """
    Mixin for services that support soft delete functionality
    """
    
    def soft_delete(self, id: Any, deleted_by: str = None) -> bool:
        """
        Soft delete record by setting del_flag = 1
        
        Args:
            id: Record ID to soft delete
            deleted_by: User performing the deletion
            
        Returns:
            True if deleted successfully, False if not found
            
        Raises:
            Exception: If deletion fails
        """
        try:
            obj = self.get_by_id(id)
            if not obj:
                return False
            
            # Check if model has del_flag field
            if not hasattr(obj, 'del_flag'):
                raise AttributeError(f"{self.model.__name__} does not support soft delete")
            
            obj.del_flag = 1
            if hasattr(obj, 'updated_by') and deleted_by:
                obj.updated_by = deleted_by
            
            self.db.commit()
            
            logger.info("Record soft deleted", 
                       model=self.model.__name__, 
                       id=id,
                       deleted_by=deleted_by)
            
            return True
            
        except Exception as e:
            self.db.rollback()
            logger.error("Failed to soft delete record", 
                        model=self.model.__name__, 
                        id=id,
                        error=str(e))
            raise
    
    def restore(self, id: Any, restored_by: str = None) -> bool:
        """
        Restore soft deleted record by setting del_flag = 0
        
        Args:
            id: Record ID to restore
            restored_by: User performing the restoration
            
        Returns:
            True if restored successfully, False if not found
            
        Raises:
            Exception: If restoration fails
        """
        try:
            obj = self.db.query(self.model).filter(self.model.id == id).first()
            if not obj or not hasattr(obj, 'del_flag') or obj.del_flag == 0:
                return False
            
            obj.del_flag = 0
            if hasattr(obj, 'updated_by') and restored_by:
                obj.updated_by = restored_by
            
            self.db.commit()
            
            logger.info("Record restored", 
                       model=self.model.__name__, 
                       id=id,
                       restored_by=restored_by)
            
            return True
            
        except Exception as e:
            self.db.rollback()
            logger.error("Failed to restore record", 
                        model=self.model.__name__, 
                        id=id,
                        error=str(e))
            raise
    
    def get_active_only(self, **kwargs) -> List[ModelType]:
        """
        Get only active (non-deleted) records
        
        Args:
            **kwargs: Arguments to pass to get_multi
            
        Returns:
            List of active model instances
        """
        filters = kwargs.get('filters', {})
        filters['del_flag'] = 0
        kwargs['filters'] = filters
        
        return self.get_multi(**kwargs)
    
    def get_deleted_only(self, **kwargs) -> List[ModelType]:
        """
        Get only deleted records
        
        Args:
            **kwargs: Arguments to pass to get_multi
            
        Returns:
            List of deleted model instances
        """
        filters = kwargs.get('filters', {})
        filters['del_flag'] = 1
        kwargs['filters'] = filters
        
        return self.get_multi(**kwargs)
