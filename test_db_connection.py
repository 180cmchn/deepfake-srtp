#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试数据库连接脚本
用于验证数据库是否可以正常连接和查询
支持 SQLite 和 MySQL
"""

import sqlite3
import pymysql
import os
from datetime import datetime
from sqlalchemy import text
from app.core.config import settings
from app.core.database import engine

def test_sqlite_connection():
    """测试 SQLite 数据库连接"""
    db_path = "deepfake_detection.db"
    
    print("=" * 50)
    print("SQLite 数据库连接测试")
    print("=" * 50)
    
    # 检查数据库文件是否存在
    if not os.path.exists(db_path):
        print(f"❌ 错误: 数据库文件 {db_path} 不存在")
        return False
    
    print(f"✅ 数据库文件存在: {db_path}")
    print(f"📁 文件大小: {os.path.getsize(db_path) / 1024:.2f} KB")
    
    try:
        # 连接数据库
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        print("✅ SQLite 连接成功")
        
        # 获取所有表
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        print(f"\n📊 数据库包含 {len(tables)} 个表:")
        for table in tables:
            print(f"   - {table[0]}")
        
        conn.close()
        return True
        
    except sqlite3.Error as e:
        print(f"❌ SQLite 连接失败: {e}")
        return False


def test_mysql_connection():
    """测试 MySQL 数据库连接 - 基于 ai-manager-plateform 方法"""
    print("=" * 50)
    print("MySQL 数据库连接测试")
    print("=" * 50)
    
    try:
        # 使用 SQLAlchemy 测试连接（与 ai-manager-plateform 一致）
        from app.core.database import test_connection_with_retry
        
        if test_connection_with_retry():
            print("✅ MySQL 连接成功")
            
            # 尝试获取数据库信息
            try:
                with engine.connect() as conn:
                    # 获取 MySQL 版本
                    result = conn.execute(text("SELECT VERSION()"))
                    version = result.fetchone()[0]
                    print(f"🔧 MySQL 版本: {version}")
                    
                    # 检查数据库是否存在
                    result = conn.execute(text("SHOW DATABASES LIKE :db_name"), {"db_name": settings.MYSQL_DATABASE})
                    db_exists = result.fetchone()
                    
                    if db_exists:
                        print(f"✅ 数据库 '{settings.MYSQL_DATABASE}' 存在")
                        
                        # 获取表信息
                        conn.execute(text(f"USE {settings.MYSQL_DATABASE}"))
                        result = conn.execute(text("SHOW TABLES"))
                        tables = result.fetchall()
                        print(f"\n📊 数据库包含 {len(tables)} 个表:")
                        for table in tables:
                            print(f"   - {table[0]}")
                    else:
                        print(f"⚠️  数据库 '{settings.MYSQL_DATABASE}' 不存在")
                        print("📝 请先创建数据库:")
                        print(f"   CREATE DATABASE {settings.MYSQL_DATABASE} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;")
                        
            except Exception as e:
                print(f"⚠️  无法获取数据库详细信息: {e}")
            
            return True
        else:
            print("❌ MySQL 连接失败")
            return False
        
    except Exception as e:
        print(f"❌ MySQL 连接测试失败: {e}")
        print("\n🔧 请检查:")
        print("   1. MySQL 服务是否启动")
        print("   2. 连接参数是否正确")
        print("   3. 用户是否有权限访问")
        print("   4. 防火墙设置是否阻止连接")
        return False


def test_database_connection():
    """测试当前配置的数据库连接"""
    print("=" * 50)
    print("当前数据库配置测试")
    print("=" * 50)
    
    print(f"🔗 数据库 URL: {settings.DATABASE_URL}")
    
    if settings.DATABASE_URL.startswith("sqlite"):
        return test_sqlite_connection()
    elif settings.DATABASE_URL.startswith("mysql+pymysql"):
        return test_mysql_connection()
    else:
        print("❌ 不支持的数据库类型")
        return False

def get_database_info():
    """获取数据库详细信息"""
    db_path = "deepfake_detection.db"
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 获取数据库版本信息
        cursor.execute("SELECT sqlite_version()")
        version = cursor.fetchone()[0]
        print(f"🔧 SQLite 版本: {version}")
        
        # 获取数据库页面大小
        cursor.execute("PRAGMA page_size")
        page_size = cursor.fetchone()[0]
        
        # 获取数据库页面数量
        cursor.execute("PRAGMA page_count")
        page_count = cursor.fetchone()[0]
        
        print(f"💾 数据库信息:")
        print(f"   页面大小: {page_size} 字节")
        print(f"   页面数量: {page_count}")
        print(f"   估计大小: {(page_size * page_count) / 1024:.2f} KB")
        
        conn.close()
        
    except sqlite3.Error as e:
        print(f"❌ 获取数据库信息失败: {e}")

if __name__ == "__main__":
    success = test_database_connection()
    if success:
        get_database_info()
        print(f"\n⏰ 测试完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\n📝 现在您可以在 VSCode 中安装数据库插件来可视化查看这些数据！")
    else:
        print("\n❌ 请检查数据库文件是否存在且未损坏")
