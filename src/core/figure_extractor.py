"""
Extract figures from PDF as images.

From paper: "capture screenshots for figures and tables"
"""

import pypdf
from pdf2image import convert_from_path
from pathlib import Path
from PIL import Image
import re
from typing import List, Dict
import logging

logger = logging.getLogger("spec2rtl.figure_extractor")


class FigureExtractor:
    """Extract figures from PDF as images."""
    
    def __init__(self, pdf_path: str, output_dir: str = "data/processed/figures"):
        """
        Initialize figure extractor.
        
        Args:
            pdf_path: Path to PDF
            output_dir: Where to save figure images
        """
        self.pdf_path = Path(pdf_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.figures = []
    
    def extract_figure_locations(self) -> List[Dict]:
        """
        Find all figure references in PDF text.
        
        Returns:
            List of dicts with:
                - figure_id: "Figure 1", "Figure 2", etc.
                - caption: Figure caption text
                    - page: Page number where figure appears
            """
        figures = []
        
        logger.info(f"Scanning PDF for figure references: {self.pdf_path}")
        
        with open(self.pdf_path, "rb") as file:
            reader = pypdf.PdfReader(file)
            
            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text()
                
                # Skip early pages (cover, table of contents, etc.)
                # Table of figures is usually in first 10 pages
                if page_num < 10:
                    # Check if this looks like a table of contents/figures
                    if "Table of Figures" in text or "List of Figures" in text:
                        logger.debug(f"Skipping page {page_num} (table of figures)")
                        continue
                    
                    # Skip if page has many figure references (likely TOC)
                    figure_count = text.count("Figure ")
                    if figure_count > 5:
                        logger.debug(f"Skipping page {page_num} (too many figure refs: {figure_count})")
                        continue
            
                # Find figure references: "Figure 1. Caption text"
                # Look for figures followed by actual content (not just page numbers)
                pattern = r'Figure\s+(\d+)\.\s+([A-Za-z][^\n]{10,200})'
                matches = re.finditer(pattern, text)
                
                for match in matches:
                    fig_num = match.group(1)
                    caption = match.group(2).strip()
                    
                    # Clean caption
                    caption = ' '.join(caption.split())
                    
                    # Skip if caption ends with just dots and numbers (TOC pattern)
                    if re.search(r'\.{3,}\s*\d+\s*$', caption):
                        logger.debug(f"Skipping Figure {fig_num} on page {page_num} (TOC pattern)")
                        continue
                    
                    # Truncate very long captions
                    if len(caption) > 200:
                        caption = caption[:197] + "..."
                    
                    figures.append({
                        "figure_id": f"Figure {fig_num}",
                        "figure_number": int(fig_num),
                        "caption": caption,
                        "page": page_num
                    })
                    
                    logger.debug(f"Found Figure {fig_num} on page {page_num}: {caption[:60]}...")
        
        # Remove duplicates (keep the one that appears latest in document)
        # Usually the real figure, not the TOC entry
        seen = {}
        for fig in figures:
            fig_num = fig['figure_number']
            if fig_num not in seen or fig['page'] > seen[fig_num]['page']:
                seen[fig_num] = fig
        
        unique_figures = list(seen.values())
        self.figures = sorted(unique_figures, key=lambda x: x['figure_number'])
        
        logger.info(f"Found {len(self.figures)} unique figures across {len(set(f['page'] for f in self.figures))} pages")
        
        return self.figures
    
    def extract_figure_images(self, dpi: int = 150) -> Dict[str, str]:
        """
        Convert PDF pages with figures to images.
        
        Note: Currently saves full pages. Future improvement: crop to actual figure regions.
        
        Args:
            dpi: Resolution for image conversion (150 is good balance of quality/size)
            
        Returns:
            Dict mapping figure_id to image file path
        """
        if not self.figures:
            self.extract_figure_locations()
        
        if not self.figures:
            logger.warning("No figures found to extract")
            return {}
        
        figure_images = {}
        
        # Get unique pages that contain figures
        pages_with_figures = sorted(set(fig['page'] for fig in self.figures))
        
        logger.info(f"Converting {len(pages_with_figures)} pages to images (DPI={dpi})")
        
        try:
            # Convert those pages to images
            images = convert_from_path(
                str(self.pdf_path),
                dpi=dpi,
                first_page=min(pages_with_figures),
                last_page=max(pages_with_figures)
            )
            
            logger.info(f"Converted {len(images)} pages successfully")
            
            # Create a mapping of page_num -> image
            page_to_image = {}
            for i, page_num in enumerate(pages_with_figures):
                if i < len(images):
                    page_to_image[page_num] = images[i]
            
            # Save images for each figure (all figures on same page get same image)
            for fig in self.figures:
                page_num = fig['page']
                fig_num = fig['figure_number']
                
                if page_num in page_to_image:
                    img = page_to_image[page_num]
                    
                    # Save figure image (note: same page = same image for now)
                    output_path = self.output_dir / f"figure_{fig_num}_page_{page_num}.png"
                    img.save(output_path, "PNG")
                    
                    figure_images[fig['figure_id']] = str(output_path)
                    logger.info(f"✓ Saved {fig['figure_id']} (page {page_num}) to {output_path.name}")
                else:
                    logger.warning(f"Could not extract image for {fig['figure_id']} (page {page_num})")
        
        except Exception as e:
            logger.error(f"Failed to convert PDF pages to images: {e}")
            logger.error("Make sure poppler is installed (needed by pdf2image)")
            return {}
        
        return figure_images
    
    def get_figures_for_section(self, section_id: str, section_content: str) -> List[Dict]:
        """
        Find which figures are referenced in a specific section.
        
        Args:
            section_id: Section identifier (e.g., "5.1.1")
            section_content: Full text of the section
            
        Returns:
            List of figure dicts that are mentioned in this section
        """
        if not self.figures:
            self.extract_figure_locations()
        
        mentioned_figures = []
        
        for fig in self.figures:
            # Check if figure is mentioned in section content
            if fig['figure_id'] in section_content or f"Fig. {fig['figure_number']}" in section_content:
                mentioned_figures.append(fig)
        
        return mentioned_figures


def extract_all_figures(pdf_path: str, extract_images: bool = True) -> Dict:
    """
    Convenience function to extract all figures.
    
    Args:
        pdf_path: Path to PDF file
        extract_images: If True, extract images. If False, just get metadata.
    
    Returns:
        Dict with:
            - figures: List of figure metadata
            - images: Dict mapping figure_id to image path
            - extractor: FigureExtractor instance for further queries
    """
    extractor = FigureExtractor(pdf_path)
    
    figures = extractor.extract_figure_locations()
    
    images = {}
    if extract_images:
        images = extractor.extract_figure_images()
    
    return {
        "figures": figures,
        "images": images,
        "extractor": extractor  # Return extractor for section queries
    }


if __name__ == "__main__":
    """Test figure extraction."""
    import sys
    from pathlib import Path
    
    # Add parent directory to path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    from src.core.logging_config import setup_logging
    
    setup_logging()
    
    print("\n🔍 Testing Figure Extraction\n")
    
    # Extract all figures with images
    result = extract_all_figures("data/input/specs/riscv-spec-20191213.pdf", extract_images=True)
    
    print(f"\n✅ Found {len(result['figures'])} figures")
    print(f"✅ Saved {len(result['images'])} images\n")
    
    for fig in result['figures']:
        image_path = result['images'].get(fig['figure_id'], 'Not extracted')
        print(f"{fig['figure_id']:12s} (page {fig['page']:2d}): {fig['caption'][:60]}")
        print(f"             Image: {image_path}")
        print()