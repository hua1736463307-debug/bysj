"""毕业版医学图像训推平台后端（v1）。

模块导入不得建目录、连数据库或加载 GPU；所有副作用集中在
``app.main.create_app`` 的 lifespan 与显式服务调用中。
"""

__all__ = ["__version__"]

__version__ = "1.0.0"
