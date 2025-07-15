"""Document loader module for handling various file formats."""

from pathlib import Path
from typing import List, Tuple

import PyPDF2
from PyPDF2.errors import PdfReadError


class DocumentLoader:
    """Handles loading documents from files and directories."""

    SUPPORTED_EXTENSIONS = {'.txt', '.pdf', '.md'}

    def __init__(self):
        self.loaded_documents = []

    def load_path(self, path: str, recursive: bool = False) -> List[Tuple[str, str]]:
        """Load documents from a file or directory path.
        Returns list of (name, content) tuples."""
        path_obj = Path(path)

        if path_obj.is_file():
            return [self._load_file(path_obj)]
        if path_obj.is_dir():
            return self._load_directory(path_obj, recursive)
        raise ValueError(f"Path does not exist: {path}")

    def _load_file(self, file_path: Path) -> Tuple[str, str]:
        """Load a single file."""
        if file_path.suffix == '.pdf':
            content = self._extract_pdf_text(file_path)
        elif file_path.suffix in {'.txt', '.md'}:
            content = file_path.read_text(encoding='utf-8')
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")

        # Create a meaningful name from the file path
        name = file_path.stem.replace('_', ' ').title()
        return (name, content)

    def _load_directory(self, dir_path: Path, recursive: bool) -> List[Tuple[str, str]]:
        """Load all supported documents from a directory."""
        pattern = '**/*' if recursive else '*'
        documents = []

        for file_path in dir_path.glob(pattern):
            if file_path.is_file() and file_path.suffix in self.SUPPORTED_EXTENSIONS:
                try:
                    documents.append(self._load_file(file_path))
                    print(f"Loaded: {file_path}")
                except (ValueError, OSError, PdfReadError) as e:
                    print(f"Error loading {file_path}: {e}")

        return documents

    def _extract_pdf_text(self, pdf_path: Path) -> str:
        """Extract text from PDF file."""
        text = []
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text.append(page.extract_text())
        return '\n'.join(text)

    def get_supported_extensions(self) -> set:
        """Get the set of supported file extensions."""
        return self.SUPPORTED_EXTENSIONS.copy()
