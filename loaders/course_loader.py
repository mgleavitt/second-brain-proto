"""Course module loader for organizing and loading educational materials.

This module provides functionality to load course materials organized by modules,
including transcripts and papers from a structured directory hierarchy.
"""

from pathlib import Path
from typing import Dict, List

class CourseModuleLoader:  # pylint: disable=too-few-public-methods
    """Specialized loader for course materials organized by modules."""

    def __init__(self, course_root: Path):
        self.course_root = Path(course_root)
        self.modules = {}
        self.metadata = {}

    def load_course(self) -> Dict[str, List[Dict]]:
        """Load all modules from a course directory."""
        transcripts_dir = self.course_root / "transcripts"
        papers_dir = self.course_root / "papers"

        # Load transcripts by module
        if transcripts_dir.exists():
            for module_dir in sorted(transcripts_dir.iterdir()):
                if module_dir.is_dir() and module_dir.name.startswith("Module"):
                    module_num = module_dir.name.split()[-1]
                    self.modules[f"module_{module_num}"] = self._load_module(module_dir)

        # Load papers
        if papers_dir.exists():
            self.modules["papers"] = self._load_papers(papers_dir)

        return self.modules

    def _load_module(self, module_dir: Path) -> List[Dict]:
        """Load all documents from a module directory."""
        documents = []
        for file_path in sorted(module_dir.glob("*.txt")):
            doc = {
                "name": file_path.stem,
                "path": str(file_path),
                "type": "transcript",
                "module": module_dir.name
            }
            documents.append(doc)
        return documents

    def _load_papers(self, papers_dir: Path) -> List[Dict]:
        """Load papers from the papers directory."""
        documents = []
        for file_path in sorted(papers_dir.glob("*.txt")):
            doc = {
                "name": file_path.stem,
                "path": str(file_path),
                "type": "paper",
                "module": "papers"
            }
            documents.append(doc)
        return documents
