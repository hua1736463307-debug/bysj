"""集中配置。

约定（CONTRACTS C01/EXECUTION_ASSETS）：
- 固定变量名：DATABASE_URL、JWT_SECRET_KEY、DATA_ROOT、MODEL_ROOT、
  ARTIFACT_ROOT、PUBLIC_ORIGIN。
- 旧变量（DEEPVISION_PROJECT_ROOT / DEEPVISION_DATA_DIR / TRAINING_LOG_ROOT）
  只在本文件做兼容映射，不要求修改用户系统全局环境。
- 导入本模块不创建目录、不连接数据库。
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]

DEV_JWT_SECRET = "dev-only-secret-change-me-bysj-medical-0123456789"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(str(REPO_ROOT / ".env"), str(REPO_ROOT / "deploy" / ".env")),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "bysj-medical"
    app_env: str = Field(default="development", description="development|test|production")
    debug: bool = False
    api_prefix: str = "/api/v1"

    database_url: str = Field(default="", description="SQLAlchemy DSN，生产必须显式设置")
    database_echo: bool = False

    jwt_secret_key: str = ""
    jwt_algorithm: str = "HS256"
    jwt_issuer: str = "bysj-medical"
    jwt_expire_minutes: int = 1440

    public_origin: str = "http://127.0.0.1:8000"
    data_root: Path = Field(default=Path("./.data/data"))
    model_root: Path = Field(default=Path("./.data/models"))
    artifact_root: Path = Field(default=Path("./.data/artifacts"))

    bootstrap_admin_username: str = ""
    bootstrap_admin_password: str = ""

    max_image_bytes: int = 20 * 1024 * 1024
    max_upload_bytes: int = 512 * 1024 * 1024
    max_zip_files: int = 10000
    max_extract_bytes: int = 2 * 1024 * 1024 * 1024
    max_compression_ratio: int = 100
    max_image_pixels: int = 40_000_000
    max_batch_inference: int = 100

    inline_jobs: bool = Field(default=False, description="仅测试/开发：请求内执行作业")
    ml_backend: str = Field(default="auto", description="auto|ultralytics|dummy")
    worker_poll_seconds: float = 2.0
    job_lease_seconds: int = 120
    gpu_lease_seconds: int = 600
    max_pending_training_per_user: int = 5
    training_epoch_seconds: int = 0

    cors_origins: list[str] = []

    @field_validator("data_root", "model_root", "artifact_root", mode="before")
    @classmethod
    def _expand_path(cls, value):
        if value in (None, ""):
            return value
        return Path(os.path.expanduser(str(value))).resolve()

    @model_validator(mode="after")
    def _apply_legacy_aliases(self) -> "Settings":
        # 只在配置层处理历史变量，避免散落到业务代码。
        if not os.environ.get("DATA_ROOT"):
            legacy = os.environ.get("DEEPVISION_DATA_DIR") or os.environ.get("DEEPVISION_PROJECT_ROOT")
            if legacy:
                self.data_root = Path(legacy).resolve() / "data"
        if not os.environ.get("MODEL_ROOT") and os.environ.get("TRAINING_LOG_ROOT"):
            self.model_root = Path(os.environ["TRAINING_LOG_ROOT"]).resolve()
        if not self.database_url:
            self.database_url = f"sqlite:///{(REPO_ROOT / '.data' / 'bysj.db').as_posix()}"
        if not self.jwt_secret_key:
            if self.app_env in {"production"}:
                raise ValueError("JWT_SECRET_KEY 必须在生产环境显式配置且不少于 32 字符")
            self.jwt_secret_key = DEV_JWT_SECRET
        if len(self.jwt_secret_key) < 32 and self.app_env == "production":
            raise ValueError("JWT_SECRET_KEY 必须不少于 32 字符")
        return self

    @property
    def is_production(self) -> bool:
        return self.app_env.lower() in {"production", "prod"}

    def ensure_runtime_dirs(self) -> None:
        for path in (self.data_root, self.model_root, self.artifact_root):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    get_settings.cache_clear()
