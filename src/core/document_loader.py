"""
Document loader for specification PDFs with section detection and logging.

From paper: "Due to LLMs' limitations in processing PDFs, we extract text 
using PyPDF and capture screenshots for figures and tables."
"""

import pypdf
import re
import logging
from typing import Dict, List, Optional
from pathlib import Path

logger = logging.getLogger("spec2rtl.document_loader")


class DocumentLoader:
    """Loads and processes specification PDF documents."""
    
    def __init__(self, pdf_path: str):
        """
        Initialize document loader.
        
        Args:
            pdf_path: Path to PDF file
        """
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        self.reader = None
        self.text_content = None
        self.pages_text = []
    
    def load(self) -> str:
        """
        Load PDF and extract all text with page markers.
        
        Returns:
            Full document text
        """
        logger.info(f"Loading PDF: {self.pdf_path}")
        
        with open(self.pdf_path, "rb") as file:
            self.reader = pypdf.PdfReader(file)
            logger.info(f"PDF has {len(self.reader.pages)} pages")
            
            self.pages_text = []
            for page_num, page in enumerate(self.reader.pages, start=1):
                text = page.extract_text()
                if text:
                    char_count = len(text)
                    logger.debug(f"Page {page_num}: extracted {char_count} characters")
                    
                    # Log first page content sample
                    if page_num == 1:
                        logger.debug(f"Page 1 preview (first 500 chars):\n{text[:500]}")
                    
                    self.pages_text.append({
                        "page": page_num,
                        "text": text
                    })
            
            self.text_content = "\n\n".join([p["text"] for p in self.pages_text])
            logger.info(f"Total extracted: {len(self.text_content):,} characters")
        
        return self.text_content
    
    def get_sections(self) -> List[Dict]:
        """
        Split document into sections by detecting headers.
        
        Detects patterns like:
        - "1. INTRODUCTION"
        - "5.1 Cipher"
        - "Appendix A - Key Expansion Examples"
        
        Returns:
            List of section dicts with keys:
                - section_id: Section identifier (e.g., "5.1")
                - title: Section title
                - content: Section text
                - page_start: Starting page number
        """
        if self.text_content is None:
            self.load()
        
        sections = []
        
        # Regex patterns for section headers
        # Matches: "1. INTRODUCTION", "5.1 Cipher", "5.1.1 SubBytes()", etc.
        patterns = [
            r'^(\d+\.\d+\.\d+)\s+(.+?)$',  # 5.1.1 SubBytes()
            r'^(\d+\.\d+)\s+(.+?)$',       # 5.1 Cipher
            r'^(\d+)\.\s+([A-Z][A-Za-z\s]+)$',  # 1. INTRODUCTION
            r'^(Appendix [A-Z])\s*[-–]\s*(.+?)$',  # Appendix A - Examples
        ]
        
        lines = self.text_content.split('\n')
        current_section = None
        current_content = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if this line is a section header
            is_header = False
            for pattern in patterns:
                match = re.match(pattern, line)
                if match:
                    # Save previous section
                    if current_section:
                        current_section["content"] = "\n".join(current_content).strip()
                        if current_section["content"]:  # Only add if has content
                            sections.append(current_section)
                    
                    # Start new section
                    section_id = match.group(1)
                    title = match.group(2).strip()
                    
                    current_section = {
                        "section_id": section_id,
                        "title": title,
                        "content": "",
                        "page_start": self._find_page_for_text(line)
                    }
                    current_content = []
                    is_header = True
                    break
            
            # Add line to current section content
            if not is_header and current_section:
                current_content.append(line)
        
        # Don't forget last section
        if current_section:
            current_section["content"] = "\n".join(current_content).strip()
            if current_section["content"]:
                sections.append(current_section)
        
        # If no sections detected, return full document as one section
        if not sections:
            logger.warning("No sections detected, using full document as single section")
            sections = [{
                "section_id": "full_document",
                "title": "Complete Specification Document",
                "content": self.text_content,
                "page_start": 1
            }]
        else:
            logger.info(f"Detected {len(sections)} sections")
            for sec in sections[:5]:  # Log first 5 sections
                logger.debug(f"  Section {sec['section_id']}: {sec['title']} (page {sec['page_start']})")
        
        return sections
    
    def _find_page_for_text(self, text: str) -> int:
        """Find which page contains the given text."""
        for page_info in self.pages_text:
            if text in page_info["text"]:
                return page_info["page"]
        return 1
    
    def get_page_count(self) -> int:
        """Get total number of pages."""
        if self.reader is None:
            with open(self.pdf_path, "rb") as file:
                self.reader = pypdf.PdfReader(file)
        return len(self.reader.pages)


def load_specification(pdf_path: str) -> Dict:
    """
    Convenience function to load a specification document.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        Dict with keys:
            - full_text: Complete document text
            - sections: List of section dicts
            - page_count: Total pages
            - file_path: Original file path
    """
    loader = DocumentLoader(pdf_path)
    
    return {
        "full_text": loader.load(),
        "sections": loader.get_sections(),
        "page_count": loader.get_page_count(),
        "file_path": str(loader.pdf_path)
    }