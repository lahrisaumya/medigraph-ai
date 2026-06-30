"""
================================================================================
FILE:    backend/core/config.py
PURPOSE: Centralised application configuration using Pydantic Settings.
         Compatible with pydantic-settings 2.x on all platforms (Windows/Mac/Linux).
================================================================================
"""

from functools import lru_cache
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables and .env file.
    All fields have sensible defaults for local development.
    """

    # ── Application ───────────────────────────────────────────────────────
    app_name:    str  = Field(default="MediGraph AI",                    alias="APP_NAME")
    app_version: str  = Field(default="1.0.0",                           alias="APP_VERSION")
    debug:       bool = Field(default=True,                              alias="DEBUG")
    secret_key:  str  = Field(default="dev-secret-change-in-production", alias="SECRET_KEY")

    # ── Gemini API ────────────────────────────────────────────────────────
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")

    # ── MongoDB Atlas ─────────────────────────────────────────────────────
    mongodb_uri:     str = Field(default="mongodb://localhost:27017", alias="MONGODB_URI")
    mongodb_db_name: str = Field(default="medigraph",                 alias="MONGODB_DB_NAME")

    # ── Neo4j Aura ────────────────────────────────────────────────────────
    neo4j_uri:      str = Field(default="bolt://localhost:7687", alias="NEO4J_URI")
    neo4j_username: str = Field(default="neo4j",                 alias="NEO4J_USERNAME")
    neo4j_password: str = Field(default="password",              alias="NEO4J_PASSWORD")

    # ── ML Model paths ────────────────────────────────────────────────────
    model_path:  str = Field(
        default="./backend/ml/saved_models/xgboost_adherence_model.pkl",
        alias="MODEL_PATH",
    )
    scaler_path: str = Field(
        default="./backend/ml/saved_models/feature_scaler.pkl",
        alias="SCALER_PATH",
    )

    # ── OpenFDA ───────────────────────────────────────────────────────────
    openfda_api_key: str = Field(default="", alias="OPENFDA_API_KEY")

    # ── CORS ──────────────────────────────────────────────────────────────
    allowed_origins: str = Field(
        default="http://localhost:3000,http://localhost:5173,http://127.0.0.1:3000",
        alias="ALLOWED_ORIGINS",
    )

    # ── Logging & limits ──────────────────────────────────────────────────
    log_level:     str = Field(default="INFO", alias="LOG_LEVEL")
    max_upload_mb: int = Field(default=10,     alias="MAX_UPLOAD_MB")

    # ── Computed properties ───────────────────────────────────────────────

    @property
    def cors_origins(self) -> List[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]

    @property
    def is_production(self) -> bool:
        return not self.debug

    @property
    def mongodb_connection_options(self) -> dict:
        return {
            "serverSelectionTimeoutMS": 5000,
            "connectTimeoutMS":         5000,
            "socketTimeoutMS":          10000,
        }

    # ── Validators ────────────────────────────────────────────────────────

    @field_validator("neo4j_uri")
    @classmethod
    def validate_neo4j_uri(cls, v: str) -> str:
        valid_prefixes = ("bolt://", "neo4j://", "neo4j+s://", "neo4j+ssc://")
        if not any(v.startswith(p) for p in valid_prefixes):
            raise ValueError(
                f"Invalid NEO4J_URI: '{v}'. Must start with bolt://, neo4j://, or neo4j+s://"
            )
        return v

    @field_validator("gemini_api_key")
    @classmethod
    def warn_missing_gemini_key(cls, v: str) -> str:
        if not v:
            import logging
            logging.getLogger(__name__).warning(
                "GEMINI_API_KEY not set — LLM features will fail. "
                "Get a free key at https://makersuite.google.com/app/apikey"
            )
        return v

    # ── Pydantic Settings config ──────────────────────────────────────────
    # Using SettingsConfigDict instead of dict literal — fixes the
    # 'encoding must be str or None, not tuple' error on Windows with
    # older pydantic-settings versions.

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        populate_by_name=True,
        extra="ignore",
        protected_namespaces=("settings_",),
    )


@lru_cache()
def get_settings() -> Settings:
    """Cached Settings singleton — constructed once, reused everywhere."""
    return Settings()


settings = get_settings()