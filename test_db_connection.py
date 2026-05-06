#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试数据库连接脚本
用于验证数据库是否可以正常连接和查询
仅支持 SQLite
"""

import sqlite3
import os
from datetime import datetime
from sqlalchemy.engine import make_url
from app.core.config import settings


def mask_database_url(raw_url: str) -> str:
    """Mask password in database URL for safe logging."""
    try:
        return make_url(raw_url).render_as_string(hide_password=True)
    except Exception:
        return raw_url


def get_sqlite_db_path() -> str:
    """Resolve sqlite file path from DATABASE_URL."""
    db_url = settings.DATABASE_URL
    prefix = "sqlite:///"
    if not db_url.startswith(prefix):
        return "deepfake_detection.db"
    path_part = db_url[len(prefix):]
    return os.path.abspath(path_part)

def test_sqlite_connection():
    """测试 SQLite 数据库连接"""
    db_path = get_sqlite_db_path()
    
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


def test_database_connection():
    """测试当前配置的数据库连接"""
    print("=" * 50)
    print("当前数据库配置测试")
    print("=" * 50)
    
    print(f"🔗 数据库 URL: {mask_database_url(settings.DATABASE_URL)}")

    if not settings.DATABASE_URL.startswith("sqlite:///"):
        print("❌ 当前项目仅支持 SQLite 数据库")
        return False

    if get_sqlite_db_path() == ":memory:":
        print("❌ test_db_connection.py 仅支持文件型 SQLite 数据库，不支持 sqlite:///:memory:")
        return False

    return test_sqlite_connection()

def get_database_info():
    """获取数据库详细信息"""
    if not settings.DATABASE_URL.startswith("sqlite"):
        print("ℹ️ 当前为非 SQLite 数据库，跳过 SQLite 文件信息展示")
        return

    db_path = get_sqlite_db_path()
    
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
        print("\n❌ 请检查 SQLite 数据库文件路径、权限以及 DATABASE_URL 配置")
