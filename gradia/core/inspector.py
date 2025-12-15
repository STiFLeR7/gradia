import os
from pathlib import Path
from typing import List, Optional

class Inspector:
    """Scans the working directory for potential dataset files."""
    
    SUPPORTED_EXTENSIONS = {'.csv', '.parquet'}

    def __init__(self, root_dir: str = "."):
        self.root_dir = Path(root_dir)

    def find_datasets(self) -> List[Path]:
        """Finds all supported dataset files in the root directory."""
        datasets = []
        for ext in self.SUPPORTED_EXTENSIONS:
            datasets.extend(self.root_dir.glob(f"*{ext}"))
        return sorted(datasets)

    def detect_split_layout(self):
        """
        Detects if proper 'train'/'val'/'test' folders exist.
        Returns a dictionary with paths or None.
        """
        layout = {}
        for split in ['train', 'val', 'validation', 'test']:
            split_dir = self.root_dir / split
            if split_dir.exists() and split_dir.is_dir():
                # Check for files inside
                files = []
                for ext in self.SUPPORTED_EXTENSIONS:
                    files.extend(list(split_dir.glob(f"*{ext}")))
                
                if files:
                    layout[split] = split_dir
        
        return layout if layout else None
