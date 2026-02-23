"""Book service with business logic."""

import hashlib
import logging
from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4

from app.domain.entities import Book
from app.domain.repositories import IBookRepository, ILLMService, IStorageService
from app.domain.services import IBookService

logger = logging.getLogger(__name__)


class BookService(IBookService):
    """Book service handling business logic."""

    def __init__(
        self,
        book_repository: IBookRepository,
        storage_service: IStorageService,
        llm_service: ILLMService,
    ):
        self.book_repository = book_repository
        self.storage_service = storage_service
        self.llm_service = llm_service

    async def create_book(
        self,
        file_content: bytes,
        filename: str,
        mime_type: str,
        title: Optional[str] = None,
        author: Optional[str] = None,
        genre: Optional[str] = None,
    ) -> Book:
        """Ingest a book — content-first.

        Metadata (title, author, genre) is resolved in priority order:

        1. Caller-supplied values — explicit user input always wins.
        2. Embedded PDF header fields (title / author / subject).
        3. LLM extraction from the opening text of the file.
        4. Fallback: filename stem as title, ``"Unknown Author"``, ``"general"``.

        The book record is persisted immediately.  Summary & embedding
        generation is handled by a **Celery background task** dispatched from
        the API layer.
        """
        # ── 1. Extract text from the file (content is the primary artifact) ──
        text_content = self._extract_text(file_content, mime_type)

        # ── 2. PDF built-in metadata (fast, no network call needed) ──────────
        pdf_meta: dict[str, str] = {}
        if mime_type == "application/pdf":
            pdf_meta = self._extract_pdf_metadata(file_content)
            if any(pdf_meta.values()):
                logger.info("PDF header metadata found: %s", pdf_meta)

        # ── 3. LLM extraction — only called when metadata is still missing ────
        llm_meta: dict[str, str] = {}
        if not title or not author:
            try:
                llm_meta = await self.llm_service.extract_book_metadata(text_content[:3000])
                logger.info("LLM extracted metadata: %s", llm_meta)
            except Exception as exc:
                logger.warning("LLM metadata extraction failed: %s", exc)

        # ── 4. Priority resolution: caller > PDF header > LLM > fallback ─────
        filename_stem = filename.rsplit(".", 1)[0].replace("_", " ").replace("-", " ").title()
        resolved_title = title or pdf_meta.get("title") or llm_meta.get("title") or filename_stem
        resolved_author = (
            author or pdf_meta.get("author") or llm_meta.get("author") or "Unknown Author"
        )
        resolved_genre = genre or llm_meta.get("genre") or "general"

        logger.info(
            "Creating book: '%s' by '%s' [%s], file: %s",
            resolved_title, resolved_author, resolved_genre, filename,
        )

        content_hash = hashlib.sha256(file_content).hexdigest()
        file_path = await self.storage_service.save_file(file_content, filename)
        logger.info("File saved: %s", file_path)

        book = Book(
            id=uuid4(),
            title=resolved_title,
            author=resolved_author,
            genre=resolved_genre,
            file_path=file_path,
            file_size=len(file_content),
            mime_type=mime_type,
            content_hash=content_hash,
            summary=None,
            embedding=None,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        created_book = await self.book_repository.create(book)
        logger.info("Book record created: %s", created_book.id)
        return created_book

    async def get_book(self, book_id: UUID) -> Optional[Book]:
        return await self.book_repository.get_by_id(book_id)

    async def list_books(self, skip: int = 0, limit: int = 100) -> list[Book]:
        return await self.book_repository.list_all(skip, limit)

    async def count_books(self) -> int:
        return await self.book_repository.count()

    async def update_book(
        self, book_id: UUID, title: str, author: str, genre: str = "general"
    ) -> Optional[Book]:
        """Update book metadata (title, author, genre)."""
        book = await self.book_repository.get_by_id(book_id)
        if not book:
            return None
        book.title = title
        book.author = author
        book.genre = genre
        book.updated_at = datetime.utcnow()
        return await self.book_repository.update(book)

    async def update_book_content(
        self,
        book_id: UUID,
        file_content: bytes,
        filename: str,
        mime_type: str,
    ) -> Optional[Book]:
        """Replace the book's file content.

        Deletes the old file, stores the new one, resets the summary so
        the background task can regenerate it.
        """
        book = await self.book_repository.get_by_id(book_id)
        if not book:
            return None

        # Remove old file
        await self.storage_service.delete_file(book.file_path)

        # Store new file
        content_hash = hashlib.sha256(file_content).hexdigest()
        file_path = await self.storage_service.save_file(file_content, filename)

        # Use targeted content update to avoid race conditions with BG tasks
        updated = await self.book_repository.update_content(
            book_id=book_id,
            file_path=file_path,
            file_size=len(file_content),
            mime_type=mime_type,
            content_hash=content_hash,
        )
        logger.info("Book content updated: %s (new file: %s)", book_id, file_path)
        return updated

    async def get_book_content(self, book_id: UUID) -> Optional[tuple[bytes, str, str]]:
        """Retrieve the stored file content for a book.

        Returns ``(file_bytes, mime_type, filename)`` or *None* if not found.
        """
        book = await self.book_repository.get_by_id(book_id)
        if not book:
            return None
        file_content = await self.storage_service.get_file(book.file_path)
        # Derive a download filename from the stored path
        filename = book.file_path.rsplit("/", 1)[-1] if "/" in book.file_path else book.file_path
        return file_content, book.mime_type, filename

    async def delete_book(self, book_id: UUID) -> bool:
        """Delete a book and its stored file."""
        logger.info(f"Deleting book: {book_id}")
        book = await self.book_repository.get_by_id(book_id)
        if not book:
            return False
        await self.storage_service.delete_file(book.file_path)
        deleted = await self.book_repository.delete(book_id)
        if deleted:
            logger.info(f"Book deleted: {book_id}")
        return deleted

    @staticmethod
    def _extract_text(file_content: bytes, mime_type: str) -> str:
        """Extract text from file content.

        Supports plain text and PDF files.  For PDFs, uses PyMuPDF (``fitz``)
        to extract text from each page.
        """
        if mime_type == "application/pdf":
            try:
                import fitz  # PyMuPDF

                doc = fitz.open(stream=file_content, filetype="pdf")
                pages = [page.get_text() for page in doc]
                doc.close()
                text = "\n".join(pages)
                return text[:10000] if text else ""
            except ImportError:
                logger.warning("PyMuPDF not installed — cannot extract PDF text")
                return ""
            except Exception as exc:
                logger.error("PDF extraction failed: %s", exc)
                return ""
        if mime_type.startswith("text/"):
            return file_content.decode("utf-8", errors="ignore")
        # Fallback: try UTF-8 decode for unknown types
        return file_content.decode("utf-8", errors="ignore")[:5000]

    @staticmethod
    def _extract_pdf_metadata(file_content: bytes) -> dict[str, str]:
        """Read title, author, and subject from the PDF document properties.

        Returns a dict with keys ``'title'``, ``'author'``, ``'genre'``
        (mapped from the PDF ``subject`` field).  Any absent field is an
        empty string.  Returns an empty dict when PyMuPDF is unavailable.
        """
        try:
            import fitz  # PyMuPDF

            doc = fitz.open(stream=file_content, filetype="pdf")
            meta = doc.metadata or {}
            doc.close()
            return {
                "title": (meta.get("title") or "").strip(),
                "author": (meta.get("author") or "").strip(),
                # PDF "subject" is the closest standard field to "genre"
                "genre": (meta.get("subject") or "").strip(),
            }
        except ImportError:
            return {}
        except Exception as exc:
            logger.warning("PDF metadata extraction failed: %s", exc)
            return {}
