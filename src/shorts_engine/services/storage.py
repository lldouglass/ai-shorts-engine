"""Asset storage service for video files and metadata."""

import hashlib
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import httpx

from shorts_engine.config import settings
from shorts_engine.logging import get_logger

logger = get_logger(__name__)


@dataclass
class StoredAsset:
    """Metadata for a stored asset."""

    id: UUID
    storage_type: str  # "local", "s3", "url"
    file_path: Path | None
    url: str | None
    file_size_bytes: int | None
    mime_type: str | None
    checksum: str | None
    metadata: dict[str, Any]
    created_at: datetime


class StorageService:
    """Service for storing and retrieving video assets.

    Supports:
    - Local file storage
    - URL references (for cloud-hosted assets)
    - Downloading from URLs to local storage
    """

    def __init__(
        self,
        base_path: Path | None = None,
        create_dirs: bool = True,
    ) -> None:
        """Initialize storage service.

        Args:
            base_path: Base directory for local storage. Defaults to ./storage
            create_dirs: Whether to create directories if they don't exist
        """
        self.base_path = base_path or Path("./storage")

        if create_dirs:
            self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        """Create storage directories."""
        dirs = [
            self.base_path / "clips",
            self.base_path / "audio",
            self.base_path / "final",
            self.base_path / "thumbnails",
            self.base_path / "temp",
        ]
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)

    def _get_subdir(self, asset_type: str) -> Path:
        """Get subdirectory for asset type."""
        type_map = {
            "scene_clip": "clips",
            "audio": "audio",
            "final_video": "final",
            "thumbnail": "thumbnails",
        }
        subdir = type_map.get(asset_type, "temp")
        return self.base_path / subdir

    def _compute_checksum(self, data: bytes) -> str:
        """Compute SHA256 checksum of data."""
        return hashlib.sha256(data).hexdigest()

    async def store_from_url(
        self,
        url: str,
        asset_type: str,
        video_job_id: UUID,
        scene_id: UUID | None = None,
        filename: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> StoredAsset:
        """Download and store an asset from a URL.

        Args:
            url: Source URL to download from
            asset_type: Type of asset (scene_clip, audio, final_video, thumbnail)
            video_job_id: Associated video job ID
            scene_id: Optional scene ID for scene-level assets
            filename: Optional custom filename
            metadata: Optional metadata to store

        Returns:
            StoredAsset with local file information
        """
        asset_id = uuid4()
        subdir = self._get_subdir(asset_type)

        # Generate filename if not provided
        if not filename:
            ext = self._guess_extension(url, asset_type)
            if scene_id:
                filename = f"{video_job_id}_{scene_id}_{asset_id.hex[:8]}{ext}"
            else:
                filename = f"{video_job_id}_{asset_id.hex[:8]}{ext}"

        file_path = subdir / filename

        logger.info(
            "storage_download_started",
            url=url[:100],
            asset_type=asset_type,
            destination=str(file_path),
        )

        try:
            async with httpx.AsyncClient(timeout=300.0, follow_redirects=True) as client:
                response = await client.get(url)
                response.raise_for_status()
                content = response.content

            # Write to file
            file_path.write_bytes(content)
            file_size = len(content)
            checksum = self._compute_checksum(content)
            mime_type = response.headers.get("content-type", self._guess_mime_type(asset_type))

            logger.info(
                "storage_download_completed",
                file_path=str(file_path),
                file_size=file_size,
            )

            return StoredAsset(
                id=asset_id,
                storage_type="local",
                file_path=file_path,
                url=url,
                file_size_bytes=file_size,
                mime_type=mime_type,
                checksum=checksum,
                metadata=metadata or {},
                created_at=datetime.now(),
            )

        except Exception as e:
            logger.error("storage_download_failed", url=url[:100], error=str(e))
            raise

    async def store_bytes(
        self,
        data: bytes,
        asset_type: str,
        video_job_id: UUID,
        scene_id: UUID | None = None,
        filename: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> StoredAsset:
        """Store raw bytes as an asset.

        Args:
            data: Raw bytes to store
            asset_type: Type of asset
            video_job_id: Associated video job ID
            scene_id: Optional scene ID
            filename: Optional custom filename
            metadata: Optional metadata

        Returns:
            StoredAsset with file information
        """
        asset_id = uuid4()
        subdir = self._get_subdir(asset_type)

        if not filename:
            ext = self._guess_extension("", asset_type)
            if scene_id:
                filename = f"{video_job_id}_{scene_id}_{asset_id.hex[:8]}{ext}"
            else:
                filename = f"{video_job_id}_{asset_id.hex[:8]}{ext}"

        file_path = subdir / filename
        file_path.write_bytes(data)

        return StoredAsset(
            id=asset_id,
            storage_type="local",
            file_path=file_path,
            url=None,
            file_size_bytes=len(data),
            mime_type=self._guess_mime_type(asset_type),
            checksum=self._compute_checksum(data),
            metadata=metadata or {},
            created_at=datetime.now(),
        )

    def store_url_reference(
        self,
        url: str,
        asset_type: str,
        video_job_id: UUID,
        scene_id: UUID | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> StoredAsset:
        """Store a URL reference without downloading.

        Args:
            url: URL of the asset
            asset_type: Type of asset
            video_job_id: Associated video job ID
            scene_id: Optional scene ID
            metadata: Optional metadata

        Returns:
            StoredAsset with URL reference
        """
        return StoredAsset(
            id=uuid4(),
            storage_type="url",
            file_path=None,
            url=url,
            file_size_bytes=None,
            mime_type=self._guess_mime_type(asset_type),
            checksum=None,
            metadata=metadata or {},
            created_at=datetime.now(),
        )

    def get_file_path(self, asset: StoredAsset) -> Path | None:
        """Get the local file path for an asset."""
        return asset.file_path

    def get_url(self, asset: StoredAsset) -> str | None:
        """Get URL for an asset (local file:// or remote URL)."""
        if asset.url:
            return asset.url
        if asset.file_path:
            return f"file://{asset.file_path.absolute()}"
        return None

    async def verify_asset(self, asset: StoredAsset) -> bool:
        """Verify an asset exists and is accessible."""
        if asset.storage_type == "local":
            if not asset.file_path:
                return False
            if not asset.file_path.exists():
                return False
            # Verify checksum if available
            if asset.checksum:
                data = asset.file_path.read_bytes()
                return self._compute_checksum(data) == asset.checksum
            return True

        elif asset.storage_type == "url":
            if not asset.url:
                return False
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.head(asset.url, follow_redirects=True)
                    return response.status_code == 200
            except Exception:
                return False

        return False

    def delete_asset(self, asset: StoredAsset) -> bool:
        """Delete a locally stored asset."""
        if asset.storage_type == "local" and asset.file_path:
            try:
                asset.file_path.unlink(missing_ok=True)
                return True
            except Exception as e:
                logger.error("storage_delete_failed", path=str(asset.file_path), error=str(e))
                return False
        return False

    def _guess_extension(self, url: str, asset_type: str) -> str:
        """Guess file extension from URL or asset type."""
        # Try to get from URL
        if url:
            path = url.split("?")[0]
            if "." in path.split("/")[-1]:
                return "." + path.split(".")[-1].lower()

        # Default by asset type
        ext_map = {
            "scene_clip": ".mp4",
            "audio": ".mp3",
            "final_video": ".mp4",
            "thumbnail": ".jpg",
        }
        return ext_map.get(asset_type, ".bin")

    def _guess_mime_type(self, asset_type: str) -> str:
        """Guess MIME type from asset type."""
        mime_map = {
            "scene_clip": "video/mp4",
            "audio": "audio/mpeg",
            "final_video": "video/mp4",
            "thumbnail": "image/jpeg",
        }
        return mime_map.get(asset_type, "application/octet-stream")
