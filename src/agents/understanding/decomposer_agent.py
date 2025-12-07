"""
Decomposer Agent - Breaks down target function into sub-functions.

From paper: "A Decomposer Agent receives both the summarized data and the 
original document, tasked with organizing the target implementation into a 
sequence of implementable sub-functions."
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage

logger = logging.getLogger("spec2rtl.decomposer_agent")


class DecomposerAgent:
    """
    Second agent in Understanding Module.
    Decomposes target function into implementable sub-functions.
    """
    
    SYSTEM_PROMPT = """You are an expert hardware design architect specializing in RTL implementation.

Your task: Decompose a high-level hardware function into implementable sub-functions.

For each decomposition, you must:
1. Identify all sub-functions needed to implement the target
2. Determine the correct implementation order
3. Specify dependencies between sub-functions
4. Provide a brief description of each sub-function's purpose

Guidelines:
- Sub-functions should be atomic (do one thing well)
- Order should follow data flow or algorithmic sequence
- Dependencies should be explicit (which sub-functions must complete before others)
- Each sub-function should map to a specific section in the specification
- Use BOTH the summaries (for overview) and original content (for implementation details)

Output format: JSON with structure:
{
  "target_function": "Name of main function",
  "sub_functions": [
    {
      "name": "SubFunctionName",
      "order": 1,
      "description": "What this sub-function does",
      "dependencies": ["PreviousFunction"],
      "spec_reference": "Section X.Y.Z"
    }
  ]
}

Be precise and technical. Focus on what matters for RTL implementation."""
    
    def __init__(self, model_client):
        """
        Initialize Decomposer Agent.
        
        Args:
            model_client: AutoGen model client (from OpenAIProvider)
        """
        self.agent = AssistantAgent(
            name="decomposer_agent",
            model_client=model_client,
            system_message=self.SYSTEM_PROMPT
        )
        
        # Track usage
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
    
    async def decompose_function(
        self,
        target_function: str,
        summaries: List[Dict],
        original_sections: List[Dict],
        target_section_id: Optional[str] = None,
        log_prompt: bool = False
    ) -> Dict:
        """
        Decompose a target function into sub-functions.
        
        Args:
            target_function: Name of function to decompose (e.g., "AES Cipher")
            summaries: List of summaries from Understanding Agent
            original_sections: Original sections from document (full content)
            target_section_id: Optional specific section to focus on (e.g., "5.1")
            log_prompt: If True, log the full prompt
            
        Returns:
            Decomposition dict with sub_functions list
        """
        logger.info(f"Decomposing function: {target_function}")
        if target_section_id:
            logger.info(f"  Focused on section: {target_section_id}")
        
        # Build prompt
        prompt = self._build_decomposition_prompt(
            target_function,
            summaries,
            original_sections,
            target_section_id
        )
        
        if log_prompt:
            logger.debug("="*70)
            logger.debug("DECOMPOSER PROMPT:")
            logger.debug("="*70)
            logger.debug(prompt)
            logger.debug("="*70)
        
        # Send to LLM
        response = await self.agent.on_messages(
            [TextMessage(content=prompt, source="user")],
            cancellation_token=None
        )
        
        response_text = response.chat_message.content
        
        # Extract usage
        if hasattr(response.chat_message, 'models_usage'):
            usage = response.chat_message.models_usage
            input_tokens = usage.prompt_tokens
            output_tokens = usage.completion_tokens
            
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            
            logger.info(f"  Tokens - Input: {input_tokens}, Output: {output_tokens}")
        
        logger.debug(f"Received response ({len(response_text)} chars)")
        
        # Parse JSON response
        decomposition = self._parse_decomposition_response(response_text)
        
        # Validate
        if not decomposition or "sub_functions" not in decomposition:
            logger.error("Failed to parse valid decomposition")
            return None
        
        logger.info(f"✓ Decomposed into {len(decomposition['sub_functions'])} sub-functions")
        
        return decomposition
    
    def _build_decomposition_prompt(
        self,
        target_function: str,
        summaries: List[Dict],
        original_sections: List[Dict],
        target_section_id: Optional[str] = None
    ) -> str:
        """Build the decomposition prompt with summaries AND original content."""
        
        # Filter to relevant sections
        if target_section_id:
            relevant_summaries = [
                s for s in summaries 
                if s['section_id'].startswith(target_section_id)
            ]
            relevant_originals = [
                s for s in original_sections
                if s['section_id'].startswith(target_section_id)
            ]
        else:
            relevant_summaries = summaries
            relevant_originals = original_sections
        
        logger.info(f"  Using {len(relevant_summaries)} summaries + {len(relevant_originals)} original sections")
        
        # Build summary context
        summary_text = []
        for s in relevant_summaries[:10]:  # Limit summaries
            summary_text.append(
                f"**Section {s['section_id']}: {s['title']}**\n{s['summary']}\n"
            )
        
        # Build original content (for detailed reference)
        original_text = []
        for s in relevant_originals[:5]:  # Limit originals (they're longer)
            # Truncate long sections
            content = s.get('content', '')[:3000]
            if len(s.get('content', '')) > 3000:
                content += "\n[... content truncated ...]"
            
            original_text.append(
                f"**Original Section {s['section_id']}: {s['title']}**\n{content}\n"
            )
        
        prompt = f"""Decompose the following hardware function into implementable sub-functions:

**Target Function:** {target_function}
{f"**Focus on Section:** {target_section_id}" if target_section_id else ""}

**HIGH-LEVEL SUMMARIES:**
{"".join(summary_text)}

---

**ORIGINAL SPECIFICATION CONTENT (for detailed reference):**
{"".join(original_text)}

---

**Your Task:**
Using BOTH the summaries (for overview) and original content (for details), decompose 
"{target_function}" into a sequence of implementable sub-functions.

Each sub-function should:
- Be atomic (single responsibility)
- Have clear inputs/outputs
- Map to a specific section in the spec
- Have defined dependencies
- Be implementable as a standalone module

Return ONLY a JSON object (no markdown, no explanation) with this structure:
{{
  "target_function": "{target_function}",
  "sub_functions": [
    {{
      "name": "SubFunctionName",
      "order": 1,
      "description": "Brief description of what this does",
      "dependencies": ["PreviousFunctionName"],
      "spec_reference": "Section X.Y.Z"
    }}
  ]
}}
"""
        
        return prompt
    
    def _parse_decomposition_response(self, response_text: str) -> Optional[Dict]:
        """
        Parse LLM response into structured decomposition.
        
        Handles:
        - Pure JSON
        - JSON wrapped in markdown code blocks
        - Malformed responses
        """
        logger.debug("Parsing decomposition response...")
        
        # Clean response
        text = response_text.strip()
        
        # Remove markdown code fences if present
        if text.startswith("```"):
            # Find the JSON content between ```
            lines = text.split('\n')
            json_lines = []
            in_code = False
            
            for line in lines:
                if line.startswith("```"):
                    in_code = not in_code
                    continue
                if in_code:
                    json_lines.append(line)
            
            text = '\n'.join(json_lines)
        
        # Try to parse JSON
        try:
            decomposition = json.loads(text)
            logger.debug(f"✓ Parsed decomposition: {len(decomposition.get('sub_functions', []))} sub-functions")
            return decomposition
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            logger.debug(f"Response text: {text[:500]}...")
            return None
    
    def _save_decomposition(self, decomposition: Dict, target_function: str):
        """Save decomposition to file."""
        output_dir = Path("data/output/decompositions")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"decomposition_{target_function.replace(' ', '_')}_{timestamp}.json"
        output_file = output_dir / filename
        
        # Add metadata
        output_data = {
            "metadata": {
                "timestamp": timestamp,
                "target_function": target_function,
                "num_sub_functions": len(decomposition.get('sub_functions', [])),
                "input_tokens": self.total_input_tokens,
                "output_tokens": self.total_output_tokens,
                "estimated_cost": self.total_cost
            },
            "decomposition": decomposition
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Decomposition saved to: {output_file}")
        return output_file
    
    def get_usage_summary(self) -> Dict:
        """Get usage statistics."""
        self.total_cost = (self.total_input_tokens / 1_000_000 * 0.15) + \
                         (self.total_output_tokens / 1_000_000 * 0.60)
        
        return {
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "estimated_cost": self.total_cost
        }


async def test_decomposer_agent():
    """Test Decomposer Agent with AES Cipher."""
    import sys
    
    # Force output to appear immediately
    print("🚀 Testing Decomposer Agent", flush=True)
    print("="*70, flush=True)
    
    try:
        from src.core.llm.openai_provider import OpenAIProvider
        from src.core.logging_config import setup_logging
        
        print("✓ Imports successful", flush=True)
        
        logger = setup_logging()
        logger.info("="*70)
        logger.info("STARTING DECOMPOSER AGENT TEST")
        logger.info("="*70)
        
        # 1. Load summaries from Understanding Agent
        print("\n📄 Loading summaries...", flush=True)
        summary_files = sorted(Path('data/output/summaries').glob('summaries_*.json'))
        
        if not summary_files:
            print("❌ No summaries found. Run Understanding Agent first!", flush=True)
            print("   Expected path: data/output/summaries/summaries_*.json", flush=True)
            return
        
        latest_summary_file = summary_files[-1]
        print(f"   Using: {latest_summary_file.name}", flush=True)
        
        with open(latest_summary_file) as f:
            summary_data = json.load(f)
        
        summaries = summary_data.get('summaries', [])
        print(f"   ✓ Loaded {len(summaries)} summaries", flush=True)
        
        # 2. Extract original sections from summaries (they contain original_content)
        print("\n📄 Extracting original content from summaries...", flush=True)
        original_sections = []
        for summary in summaries:
            original_sections.append({
                'section_id': summary['section_id'],
                'title': summary['title'],
                'content': summary.get('original_content', '')
            })
        print(f"   ✓ Extracted {len(original_sections)} original sections", flush=True)
        
        # 3. Create agent
        print("\n🤖 Creating Decomposer Agent...", flush=True)
        provider = OpenAIProvider(
            model_name="gpt-4o-mini",
            temperature=0.3
        )
        
        model_client = provider.create_model_client()
        agent = DecomposerAgent(model_client)
        print("   ✓ Agent created", flush=True)
        
        # 4. Test decomposition of AES Cipher (Section 5.1)
        print("\n🔨 Decomposing: AES Cipher (Section 5.1)", flush=True)
        print("   This may take 10-30 seconds...", flush=True)
        
        decomposition = await agent.decompose_function(
            target_function="AES Cipher",
            summaries=summaries,
            original_sections=original_sections,
            target_section_id="5.1",
            log_prompt=True
        )
        
        if decomposition:
            print("\n✅ Decomposition Complete", flush=True)
            print("="*70, flush=True)
            print(f"Target: {decomposition['target_function']}", flush=True)
            print(f"Sub-functions: {len(decomposition['sub_functions'])}", flush=True)
            print("="*70, flush=True)
            
            for sf in decomposition['sub_functions']:
                deps = f" (depends on: {', '.join(sf['dependencies'])})" if sf['dependencies'] else ""
                print(f"\n{sf['order']}. {sf['name']}{deps}", flush=True)
                print(f"   Description: {sf['description']}", flush=True)
                print(f"   Spec: {sf.get('spec_reference', 'N/A')}", flush=True)
            
            # Save
            output_file = agent._save_decomposition(decomposition, "AES_Cipher")
            print(f"\n💾 Saved to: {output_file}", flush=True)
            
            # Show usage
            usage = agent.get_usage_summary()
            print(f"\n💰 Cost: ${usage['estimated_cost']:.4f}", flush=True)
            print(f"   Input: {usage['input_tokens']:,} tokens", flush=True)
            print(f"   Output: {usage['output_tokens']:,} tokens", flush=True)
            
            print("\n" + "="*70, flush=True)
            print("✅ TEST COMPLETE", flush=True)
            print("="*70, flush=True)
        else:
            print("\n❌ Decomposition failed", flush=True)
            logger.error("DECOMPOSER AGENT TEST FAILED")
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    print("Starting decomposer_agent.py...", flush=True)
    asyncio.run(test_decomposer_agent())
    print("Finished.", flush=True)