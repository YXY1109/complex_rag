"""
DeepDoc Configuration (RAGFlow Reference)

This module contains configuration for RAGFlow DeepDoc document parsing capabilities.
"""

from pydantic import Field

from ..settings import BaseConfig


class DeepDocConfig(BaseConfig):
    """DeepDoc configuration based on RAGFlow best practices."""

    # OCR Configuration
    ocr_enabled: bool = Field(default=True, env="DEEPDOC_OCR_ENABLED")
    ocr_engine: str = Field(default="tesseract", env="DEEPDOC_OCR_ENGINE")  # tesseract, paddleocr, easyocr
    ocr_languages: list[str] = Field(default=["eng", "chi_sim"], env="DEEPDOC_OCR_LANGUAGES")
    ocr_dpi: int = Field(default=300, env="DEEPDOC_OCR_DPI")
    ocr_timeout: int = Field(default=120, env="DEEPDOC_OCR_TIMEOUT")

    # Layout Recognition
    layout_recognition_enabled: bool = Field(default=True, env="DEEPDOC_LAYOUT_ENABLED")
    layout_model_path: str = Field(default="models/layout", env="DEEPDOC_LAYOUT_MODEL_PATH")
    layout_confidence_threshold: float = Field(default=0.8, env="DEEPDOC_LAYOUT_CONFIDENCE_THRESHOLD")
    layout_max_workers: int = Field(default=4, env="DEEPDOC_LAYOUT_MAX_WORKERS")

    # Table Structure Recognition
    table_recognition_enabled: bool = Field(default=True, env="DEEPDOC_TABLE_ENABLED")
    table_model_path: str = Field(default="models/table", env="DEEPDOC_TABLE_MODEL_PATH")
    table_confidence_threshold: float = Field(default=0.7, env="DEEPDOC_TABLE_CONFIDENCE_THRESHOLD")
    table_max_size: int = Field(default=20, env="DEEPDOC_TABLE_MAX_SIZE")  # max 20x20 cells

    # Image Processing
    image_preprocessing_enabled: bool = Field(default=True, env="DEEPDOC_IMAGE_PREPROCESSING_ENABLED")
    image_resize_max_dim: int = Field(default=2000, env="DEEPDOC_IMAGE_RESIZE_MAX_DIM")
    image_quality_threshold: float = Field(default=0.5, env="DEEPDOC_IMAGE_QUALITY_THRESHOLD")
    image_format: str = Field(default="RGB", env="DEEPDOC_IMAGE_FORMAT")

    # Document Processing
    chunk_size: int = Field(default=1024, env="DEEPDOC_CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, env="DEEPDOC_CHUNK_OVERLAP")
    min_chunk_size: int = Field(default=100, env="DEEPDOC_MIN_CHUNK_SIZE")
    max_chunk_size: int = Field(default=4096, env="DEEPDOC_MAX_CHUNK_SIZE")

    # PDF Processing
    pdf_extract_images: bool = Field(default=True, env="DEEPDOC_PDF_EXTRACT_IMAGES")
    pdf_extract_tables: bool = Field(default=True, env="DEEPDOC_PDF_EXTRACT_TABLES")
    pdf_extract_text: bool = Field(default=True, env="DEEPDOC_PDF_EXTRACT_TEXT")
    pdf_password_protected: bool = Field(default=False, env="DEEPDOC_PDF_PASSWORD_PROTECTED")
    pdf_password: str = Field(default="", env="DEEPDOC_PDF_PASSWORD")

    # Multi-modal Processing
    multimodal_enabled: bool = Field(default=True, env="DEEPDOC_MULTIMODAL_ENABLED")
    image_description_model: str = Field(default="gpt-4-vision-preview", env="DEEPDOC_IMAGE_DESCRIPTION_MODEL")
    image_description_max_length: int = Field(default=200, env="DEEPDOC_IMAGE_DESCRIPTION_MAX_LENGTH")
    image_description_enabled: bool = Field(default=False, env="DEEPDOC_IMAGE_DESCRIPTION_ENABLED")

    # Text Processing
    text_cleaning_enabled: bool = Field(default=True, env="DEEPDOC_TEXT_CLEANING_ENABLED")
    remove_headers_footers: bool = Field(default=True, env="DEEPDOC_REMOVE_HEADERS_FOOTERS")
    remove_page_numbers: bool = Field(default=True, env="DEEPDOC_REMOVE_PAGE_NUMBERS")
    normalize_whitespace: bool = Field(default=True, env="DEEPDOC_NORMALIZE_WHITESPACE")
    remove_special_chars: bool = Field(default=False, env="DEEPDOC_REMOVE_SPECIAL_CHARS")

    # Language Detection
    language_detection_enabled: bool = Field(default=True, env="DEEPDOC_LANGUAGE_DETECTION_ENABLED")
    default_language: str = Field(default="auto", env="DEEPDOC_DEFAULT_LANGUAGE")
    supported_languages: list[str] = Field(
        default=["en", "zh", "ja", "ko", "es", "fr", "de", "it", "pt", "ru"],
        env="DEEPDOC_SUPPORTED_LANGUAGES"
    )

    # Quality Control
    quality_check_enabled: bool = Field(default=True, env="DEEPDOC_QUALITY_CHECK_ENABLED")
    min_text_density: float = Field(default=0.1, env="DEEPDOC_MIN_TEXT_DENSITY")
    max_blank_pages: int = Field(default=3, env="DEEPDOC_MAX_BLANK_PAGES")
    min_text_confidence: float = Field(default=0.5, env="DEEPDOC_MIN_TEXT_CONFIDENCE")

    # Performance Settings
    max_concurrent_pages: int = Field(default=10, env="DEEPDOC_MAX_CONCURRENT_PAGES")
    processing_timeout: int = Field(default=600, env="DEEPDOC_PROCESSING_TIMEOUT")  # 10 minutes
    memory_limit_mb: int = Field(default=4096, env="DEEPDOC_MEMORY_LIMIT_MB")

    # Cache Settings
    cache_enabled: bool = Field(default=True, env="DEEPDOC_CACHE_ENABLED")
    cache_ttl: int = Field(default=86400, env="DEEPDOC_CACHE_TTL")  # 24 hours
    cache_dir: str = Field(default="cache/deepdoc", env="DEEPDOC_CACHE_DIR")

    # File Size Limits
    max_file_size_mb: int = Field(default=100, env="DEEPDOC_MAX_FILE_SIZE_MB")
    max_pages: int = Field(default=1000, env="DEEPDOC_MAX_PAGES")
    max_images_per_page: int = Field(default=50, env="DEEPDOC_MAX_IMAGES_PER_PAGE")

    # Output Format
    output_format: str = Field(default="markdown", env="DEEPDOC_OUTPUT_FORMAT")  # markdown, json, text
    include_metadata: bool = Field(default=True, env="DEEPDOC_INCLUDE_METADATA")
    include_confidence: bool = Field(default=False, env="DEEPDOC_INCLUDE_CONFIDENCE")
    include_coordinates: bool = Field(default=False, env="DEEPDOC_INCLUDE_COORDINATES")

    # Error Handling
    continue_on_error: bool = Field(default=True, env="DEEPDOC_CONTINUE_ON_ERROR")
    retry_failed_pages: bool = Field(default=True, env="DEEPDOC_RETRY_FAILED_PAGES")
    max_retries: int = Field(default=3, env="DEEPDOC_MAX_RETRIES")
    retry_delay: int = Field(default=5, env="DEEPDOC_RETRY_DELAY")  # seconds

    def get_ocr_config(self) -> dict:
        """Get OCR configuration."""
        return {
            "engine": self.ocr_engine,
            "languages": self.ocr_languages,
            "dpi": self.ocr_dpi,
            "timeout": self.ocr_timeout,
            "enabled": self.ocr_enabled,
        }

    def get_layout_config(self) -> dict:
        """Get layout recognition configuration."""
        return {
            "enabled": self.layout_recognition_enabled,
            "model_path": self.layout_model_path,
            "confidence_threshold": self.layout_confidence_threshold,
            "max_workers": self.layout_max_workers,
        }

    def get_table_config(self) -> dict:
        """Get table recognition configuration."""
        return {
            "enabled": self.table_recognition_enabled,
            "model_path": self.table_model_path,
            "confidence_threshold": self.table_confidence_threshold,
            "max_size": self.table_max_size,
        }

    def get_processing_config(self) -> dict:
        """Get document processing configuration."""
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "min_chunk_size": self.min_chunk_size,
            "max_chunk_size": self.max_chunk_size,
            "text_cleaning": {
                "enabled": self.text_cleaning_enabled,
                "remove_headers_footers": self.remove_headers_footers,
                "remove_page_numbers": self.remove_page_numbers,
                "normalize_whitespace": self.normalize_whitespace,
                "remove_special_chars": self.remove_special_chars,
            },
            "quality_check": {
                "enabled": self.quality_check_enabled,
                "min_text_density": self.min_text_density,
                "max_blank_pages": self.max_blank_pages,
                "min_text_confidence": self.min_text_confidence,
            },
        }

    def get_pdf_config(self) -> dict:
        """Get PDF processing configuration."""
        return {
            "extract_images": self.pdf_extract_images,
            "extract_tables": self.pdf_extract_tables,
            "extract_text": self.pdf_extract_text,
            "password_protected": self.pdf_password_protected,
            "password": self.pdf_password,
        }

    def get_multimodal_config(self) -> dict:
        """Get multi-modal processing configuration."""
        return {
            "enabled": self.multimodal_enabled,
            "image_description": {
                "enabled": self.image_description_enabled,
                "model": self.image_description_model,
                "max_length": self.image_description_max_length,
            },
            "image_processing": {
                "enabled": self.image_preprocessing_enabled,
                "resize_max_dim": self.image_resize_max_dim,
                "quality_threshold": self.image_quality_threshold,
                "format": self.image_format,
            },
        }


# Global DeepDoc configuration instance
deepdoc_config = DeepDocConfig()