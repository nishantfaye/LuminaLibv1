"""Local file storage implementation."""
import hashlib
import logging
import os
from pathlib import Path

import aiofiles

from app.domain.repositories import IStorageService

logger = logging.getLogger(__name__)


class LocalStorageService(IStorageService):
    """Content-addressable file storage on the local filesystem."""
    
    def __init__(self, base_path: str = "./storage"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    async def save_file(self, file_content: bytes, filename: str) -> str:
        try:
            file_hash = hashlib.sha256(file_content).hexdigest()
            subdir = file_hash[:2]
            storage_dir = self.base_path / subdir
            storage_dir.mkdir(parents=True, exist_ok=True)
            
            file_path = storage_dir / f"{file_hash}_{filename}"
            
            async with aiofiles.open(file_path, "wb") as f:
                await f.write(file_content)
            
            logger.info(f"File saved: {file_path.name}, size: {len(file_content)} bytes")
            return str(file_path.relative_to(self.base_path))
            
        except Exception as e:
            logger.error(f"Failed to save file {filename}: {str(e)}", exc_info=True)
            raise
    
    async def get_file(self, file_path: str) -> bytes:
        try:
            full_path = self.base_path / file_path
            async with aiofiles.open(full_path, "rb") as f:
                content = await f.read()
            logger.debug(f"File retrieved: {file_path}, size: {len(content)} bytes")
            return content
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Failed to read file {file_path}: {str(e)}", exc_info=True)
            raise
    
    async def delete_file(self, file_path: str) -> bool:
        full_path = self.base_path / file_path
        try:
            os.remove(full_path)
            logger.info(f"File deleted: {file_path}")
            return True
        except FileNotFoundError:
            logger.warning(f"File not found for deletion: {file_path}")
            return False
        except Exception as e:
            logger.error(f"Failed to delete file {file_path}: {str(e)}", exc_info=True)
            raise
