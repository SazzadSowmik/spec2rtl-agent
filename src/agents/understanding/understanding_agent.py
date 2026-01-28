"""
Understanding Agent - Summarizes specification sections with vision support.

From paper: "Due to LLMs' limitations in processing PDFs, we extract text 
using PyPDF and capture screenshots for figures and tables. All extracted 
data is compiled and fed into the LLM."
"""

import asyncio
import logging
import base64
from typing import Dict, List, Optional
from pathlib import Path
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage

logger = logging.getLogger("spec2rtl.understanding_agent")


class UnderstandingAgent:
    """
    First agent in Understanding Module.
    Generates concise summaries using text + figure images.
    """
    
    SYSTEM_PROMPT = """You are an expert technical documentation analyst specializing in hardware specifications.

Your task: Summarize specification document sections concisely and accurately.

For each section, provide:
1. A 2-3 sentence summary of the main purpose
2. Key technical concepts (3-5 bullet points)
3. Important terminology
4. Dependencies on other sections (if any)

When figures/diagrams/tables are provided as images, carefully analyze them and incorporate 
visual information into your summary. Pay special attention to:
- State transformation diagrams
- Algorithm pseudocode
- Lookup tables (S-boxes, etc.)
- Data flow diagrams

Keep summaries clear, technical, and focused on what matters for RTL implementation."""
    
    def __init__(self, model_client, figures_dir: str = "data/processed/figures"):
        """
        Initialize Understanding Agent.
        
        Args:
            model_client: AutoGen model client (should support vision, e.g., gpt-4o)
            figures_dir: Directory containing figure images
        """
        self.agent = AssistantAgent(
            name="understanding_agent",
            model_client=model_client,
            system_message=self.SYSTEM_PROMPT
        )
        
        self.figures_dir = Path(figures_dir)
        self.available_figures = self._scan_figures()
        
        # Track token usage
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        
        logger.info(f"Understanding Agent initialized with {len(self.available_figures)} figures")
    
    def _scan_figures(self) -> Dict[str, str]:
        """
        Scan figures directory for available images.
        
        Returns:
            Dict mapping figure_id (e.g., "figure_1") to file path
        """
        figures = {}
        
        if not self.figures_dir.exists():
            logger.warning(f"Figures directory not found: {self.figures_dir}")
            return figures
        
        # Scan for figure_N.png, table_N.png
        for img_file in self.figures_dir.glob("*.png"):
            fig_id = img_file.stem  # e.g., "figure_1", "table_2"
            figures[fig_id] = str(img_file)
            logger.debug(f"Found {fig_id}: {img_file.name}")
        
        return figures
    
    def _encode_image_base64(self, image_path: str) -> str:
        """Encode image to base64 for OpenAI vision API."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')
    
    def _find_figures_in_section(self, section_content: str) -> List[str]:
        """
        Find which figures are mentioned in section content.
        
        Args:
            section_content: Text content of section
            
        Returns:
            List of figure_ids mentioned (e.g., ["figure_1", "figure_5"])
        """
        mentioned = []
        
        for fig_id in self.available_figures.keys():
            # Check for "Figure 1", "Fig. 1", "figure 1", etc.
            fig_num = fig_id.replace("figure_", "").replace("table_", "")
            
            if fig_id.startswith("figure"):
                patterns = [f"Figure {fig_num}", f"Fig. {fig_num}", f"figure {fig_num}"]
            else:  # table
                patterns = [f"Table {fig_num}", f"table {fig_num}"]
            
            if any(pattern in section_content for pattern in patterns):
                mentioned.append(fig_id)
        
        return mentioned
    
    def _build_vision_prompt(
        self, 
        section_id: str,
        title: str,
        content: str,
        figure_ids: List[str]
    ) -> str:
        """
        Build prompt with text + figure references.
        
        Note: For now, we mention figures in text. 
        Full vision integration requires OpenAI client with image support.
        """
        content_snippet = content

        # COSTING MUCH? UNCOMMENT TO TRIM
        # content_snippet = content[:5000]
        # if len(content) > 5000:
        #     content_snippet += "\n\n[... content truncated ...]"
        
        prompt = f"""Summarize this specification section:

**Section:** {section_id}
**Title:** {title}

**Content:**
{content_snippet}
"""
        
        if figure_ids:
            prompt += f"\n\n**Note:** This section references the following figures/tables: {', '.join(figure_ids)}"
            prompt += "\nImages of these figures are provided. Analyze them carefully and incorporate key visual information into your summary."
        
        prompt += """

Provide a clear, technical summary focusing on:
- Main purpose and functionality
- Key implementation details (including information from figures/tables)
- Important terminology
- Any dependencies or prerequisites"""
        
        return prompt
    
    async def summarize_section(
        self, 
        section: Dict,
        log_prompt: bool = False
    ) -> Dict:
        """
        Summarize one specification section with vision support.
        
        Args:
            section: Dict with keys: section_id, title, content, page_start
            log_prompt: If True, log the full prompt
            
        Returns:
            Summary dict
        """
        section_id = section.get("section_id", "unknown")
        title = section.get("title", "Untitled")
        content = section.get("content", "")
        
        logger.info(f"Processing section: {section_id} - {title}")
        
        # Find relevant figures
        figure_ids = self._find_figures_in_section(content)
        
        if figure_ids:
            logger.info(f"  Found {len(figure_ids)} figures: {', '.join(figure_ids)}")
        
        # Build prompt
        prompt = self._build_vision_prompt(section_id, title, content, figure_ids)
        
        if log_prompt:
            logger.debug("="*70)
            logger.debug("PROMPT SENT TO LLM:")
            logger.debug("="*70)
            logger.debug(prompt)
            if figure_ids:
                logger.debug(f"\nFigures to be sent: {figure_ids}")
            logger.debug("="*70)
        
        # TODO: For full vision support, we need to use OpenAI client directly
        # AutoGen 0.4 multimodal support is still evolving
        # For now, we send text + figure references
        
        response = await self.agent.on_messages(
            [TextMessage(content=prompt, source="user")],
            cancellation_token=None
        )
        
        summary_text = response.chat_message.content
        
       # Extract token usage from response
        usage_info = {
            'input_tokens': 0,
            'output_tokens': 0,
            'total_tokens': 0
        }

        # FIX: Use models_usage instead of usage
        if hasattr(response.chat_message, 'models_usage'):
            usage = response.chat_message.models_usage
            usage_info['input_tokens'] = usage.prompt_tokens
            usage_info['output_tokens'] = usage.completion_tokens
            usage_info['total_tokens'] = usage.prompt_tokens + usage.completion_tokens
            
            # Accumulate totals
            self.total_input_tokens += usage_info['input_tokens']
            self.total_output_tokens += usage_info['output_tokens']
            
            logger.info(f"  Tokens - Input: {usage_info['input_tokens']}, Output: {usage_info['output_tokens']}")
        else:
            logger.warning("❌ No models_usage found in response")

        logger.debug(f"Received response ({len(summary_text)} chars)")

        # BUILD AND RETURN RESULT
        result = {
            "section_id": section_id,
            "title": title,
            "summary": summary_text,
            "original_content": content,
            "usage": usage_info,
            "figures_referenced": figure_ids
        }
        
        logger.info(f"✓ Completed: {section_id}")
        
        return result
    
    async def process_document(self, sections: List[Dict]) -> List[Dict]:
        """
        Process all sections in a document.
        
        Args:
            sections: List of section dicts
            
        Returns:
            List of summary dicts
        """
        logger.info(f"Starting document processing: {len(sections)} sections")
        logger.info(f"Available figures: {len(self.available_figures)}")
        
        summaries = []
        
        # Sections to log full prompts for
        # important_sections = ["5.1.1", "5.1.2", "5.1.4", "5.2"]
        
        for i, section in enumerate(sections, 1):
            section_id = section.get("section_id", f"section_{i}")
            title = section.get("title", "Untitled")
            
            logger.info(f"[{i}/{len(sections)}] {section_id} - {title}")
            
            # Log prompts for first section + important sections
            # log_prompt = (i == 1) or (section_id in important_sections)
            log_prompt = True
            
            try:
                summary = await self.summarize_section(section, log_prompt=log_prompt)
                
                # Validate result
                if summary is None:
                    logger.error(f"❌ Section {section_id} returned None, skipping")
                    continue
                
                if not isinstance(summary, dict):
                    logger.error(f"❌ Section {section_id} returned non-dict: {type(summary)}, skipping")
                    continue
                
                summaries.append(summary)
                
            except Exception as e:
                logger.error(f"❌ Error processing section {section_id}: {e}", exc_info=True)
                continue
        
        logger.info(f"✅ Document processing complete: {len(summaries)} summaries")
        logger.info(f"Total tokens - Input: {self.total_input_tokens:,}, Output: {self.total_output_tokens:,}")
        
        # Calculate cost (gpt-4o-mini: $0.15/1M input, $0.60/1M output)
        self.total_cost = (self.total_input_tokens / 1_000_000 * 0.15) + (self.total_output_tokens / 1_000_000 * 0.60)
        logger.info(f"Estimated cost: ${self.total_cost:.4f}")
        
        self._save_summaries(summaries)
        
        return summaries
    
    def _save_summaries(self, summaries: List[Dict]):
        """Save summaries to JSON."""
        from pathlib import Path
        import json
        from datetime import datetime

        output_dir = Path("data/output/summaries")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"summaries_{timestamp}.json"
        
        # Add metadata
        output_data = {
            "metadata": {
                "timestamp": timestamp,
                "total_sections": len(summaries),
                "total_input_tokens": self.total_input_tokens,
                "total_output_tokens": self.total_output_tokens,
                "estimated_cost": self.total_cost,
                "figures_available": len(self.available_figures)
            },
            "summaries": [
                {
                    "section_id": s["section_id"],
                    "title": s["title"],
                    "summary": s["summary"],
                    "original_content": s.get("original_content", ""),  # ← ADD THIS LINE
                    "content_length": len(s.get("original_content", "")),
                    "usage": s["usage"],
                    "figures_referenced": s["figures_referenced"]
                }
                for s in summaries
            ]
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Summaries saved to: {output_file}")
    
    def get_usage_summary(self) -> Dict:
        """Get usage statistics."""
        return {
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "estimated_cost": self.total_cost
        }


async def test_understanding_agent():
    """Test Understanding Agent with manual figures."""
    from src.core.llm.openai_provider import OpenAIProvider
    from src.core.document_loader import load_specification
    from src.core.logging_config import setup_logging
    
    logger = setup_logging()
    
    print("🚀 Testing Understanding Agent with Vision Support\n")
    logger.info("="*70)
    logger.info("STARTING UNDERSTANDING AGENT TEST")
    logger.info("="*70)
    
    # Load PDF
    pdf_path = "data/input/specs/riscv-spec-20191213.pdf"
    print(f"📄 Loading: {pdf_path}")
    
    spec_doc = load_specification(pdf_path)
    print(f"   Pages: {spec_doc['page_count']}")
    print(f"   Sections: {len(spec_doc['sections'])}")
    print(f"   Total chars: {len(spec_doc['full_text']):,}\n")
    
    # Create provider and agent
    provider = OpenAIProvider(
        model_name="gpt-5.2",  # Supports vision
        temperature=0.3
    )
    
    model_client = provider.create_model_client()
    agent = UnderstandingAgent(model_client)
    
    print(f"🖼️  Available figures: {len(agent.available_figures)}")
    if agent.available_figures:
        print(f"   Figures: {', '.join(sorted(agent.available_figures.keys())[:10])}...")
    else:
        print(f"   ⚠️  No figures found in data/processed/figures/")
        print(f"   Place your manual screenshots there as: figure_1.png, figure_2.png, etc.\n")
    
    # Process document
    summaries = await agent.process_document(spec_doc['sections'])
    
    print(f"\n✅ Generated {len(summaries)} summaries\n")
    
    # Display first summary with figures
    for s in summaries[:3]:
        if s.get('figures_referenced'):
            print("="*70)
            print(f"Section: {s['section_id']} - {s['title']}")
            print(f"Figures: {', '.join(s['figures_referenced'])}")
            print("="*70)
            print(f"\n{s['summary'][:300]}...\n")
            break
    
    # Show costs
    usage = agent.get_usage_summary()
    print(f"\n💰 Cost: ${usage['estimated_cost']:.4f}")
    print(f"   Input: {usage['input_tokens']:,} tokens")
    print(f"   Output: {usage['output_tokens']:,} tokens")


if __name__ == "__main__":
    asyncio.run(test_understanding_agent())