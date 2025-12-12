"""
Description Agent - Creates detailed implementation info for sub-functions.

From paper: "The Description Agent uses the original document, the summaries 
provided, and specific sub-function requirements as inputs to collate and 
format the necessary information into a structured dictionary."
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage

logger = logging.getLogger("spec2rtl.description_agent")


class DescriptionAgent:
    """
    Third agent in Understanding Module.
    Creates detailed implementation dictionaries for sub-functions.
    """
    
    SYSTEM_PROMPT = """You are an expert RTL implementation specialist.

        Your task: Create detailed implementation information for hardware sub-functions.

        For each sub-function, you must provide:
        1. **Description**: Clear explanation of what the sub-function does
        2. **Inputs**: All input parameters with types/sizes
        3. **Outputs**: All output parameters with types/sizes
        4. **Functionality**: Detailed explanation of the algorithm/operation
        5. **Steps**: Numbered, implementable steps for the algorithm
        6. **References**: Specific sections, figures, tables from the specification
        7. **Implementation Notes**: Critical details for RTL coding (data types, constraints, edge cases)

        Guidelines:
        - Be precise about data types and sizes (e.g., "4x4 array of bytes", "32-bit word")
        - Steps should be atomic and directly implementable in code
        - Reference all relevant figures, tables, and equations
        - Include edge cases and constraints
        - Focus on what an RTL engineer needs to implement this

        Output format: JSON with structure:
        {
        "sub_function": "FunctionName",
        "description": "One sentence summary",
        "inputs": ["input1 (type/size)", "input2 (type/size)"],
        "outputs": ["output1 (type/size)"],
        "functionality": "Detailed explanation of the algorithm",
        "steps": [
            "1. First step with specifics",
            "2. Second step with specifics"
        ],
        "references": ["Section X.Y.Z", "Figure N", "Table M"],
        "implementation_notes": "Critical details for implementation"
        }

        Be thorough and precise. This will be used directly for RTL code generation."""
    
    def __init__(self, model_client):
        """
        Initialize Description Agent.
        
        Args:
            model_client: AutoGen model client (from OpenAIProvider)
        """
        self.agent = AssistantAgent(
            name="description_agent",
            model_client=model_client,
            system_message=self.SYSTEM_PROMPT
        )
        
        # Track usage
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
    
    async def describe_sub_function(
        self,
        sub_function: Dict,
        summaries: List[Dict],
        original_sections: List[Dict],
        verifier_feedback: Optional[List[str]] = None,
        log_prompt: bool = False
    ) -> Dict:
        """
        Create detailed description for a sub-function.
        
        Args:
            sub_function: Dict from Decomposer with name, order, description, spec_reference
            summaries: List of summaries from Understanding Agent
            original_sections: Original sections from document
            verifier_feedback: Optional feedback from Verifier for revision
            log_prompt: If True, log the full prompt
            
        Returns:
            Detailed description dict
        """
        sub_func_name = sub_function.get('name', 'Unknown')
        spec_ref = sub_function.get('spec_reference', '')
        
        logger.info(f"Describing sub-function: {sub_func_name}")
        if spec_ref:
            logger.info(f"  Reference: {spec_ref}")
        
        # Build prompt
        prompt = self._build_description_prompt(
            sub_function,
            summaries,
            original_sections,
            verifier_feedback
        )
        
        if log_prompt:
            logger.debug("="*70)
            logger.debug("DESCRIPTION AGENT PROMPT:")
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
        description = self._parse_description_response(response_text)
        
        # Validate
        if not description:
            logger.error("Failed to parse valid description")
            return None
        
        logger.info(f"✓ Created description for: {sub_func_name}")
        
        return description
    
    def _build_description_prompt(
        self,
        sub_function: Dict,
        summaries: List[Dict],
        original_sections: List[Dict],
        verifier_feedback: Optional[List[str]] = None
    ) -> str:
        """Build the description prompt."""
        
        sub_func_name = sub_function.get('name', 'Unknown')
        sub_func_desc = sub_function.get('description', '')
        spec_ref = sub_function.get('spec_reference', '')
        
        # Find relevant sections based on spec_reference
        # Extract section ID from reference (e.g., "Section 5.1.1" -> "5.1.1")
        section_id = self._extract_section_id(spec_ref)
        
        # Get relevant summaries and original content
        relevant_summaries = []
        relevant_originals = []
        
        if section_id:
            # Exact match first
            for s in summaries:
                if s['section_id'] == section_id:
                    relevant_summaries.append(s)
            
            for s in original_sections:
                if s['section_id'] == section_id:
                    relevant_originals.append(s)
            
            # If no exact match, try prefix match
            if not relevant_summaries:
                for s in summaries:
                    if s['section_id'].startswith(section_id.split('.')[0]):
                        relevant_summaries.append(s)
                
                for s in original_sections:
                    if s['section_id'].startswith(section_id.split('.')[0]):
                        relevant_originals.append(s)
        
        logger.info(f"  Using {len(relevant_summaries)} summaries + {len(relevant_originals)} original sections")
        
        # Build summary context
        summary_text = []
        for s in relevant_summaries[:5]:
            summary_text.append(
                f"**Section {s['section_id']}: {s['title']}**\n{s['summary']}\n"
            )
        
        # Build original content
        original_text = []
        for s in relevant_originals[:3]:
            content = s.get('content', '')[:5000]  # Limit to 5000 chars
            if len(s.get('content', '')) > 5000:
                content += "\n[... content truncated ...]"
            
            original_text.append(
                f"**Original Section {s['section_id']}: {s['title']}**\n{content}\n"
            )
        
        # Build feedback section if provided
        feedback_text = ""
        if verifier_feedback:
            feedback_text = f"""
**FEEDBACK FROM VERIFIER (address these issues):**
{chr(10).join(f"- {fb}" for fb in verifier_feedback)}

Please revise your description to address the feedback above.
"""
        
        prompt = f"""Create a detailed implementation description for the following hardware sub-function:

**Sub-function Name:** {sub_func_name}
**Brief Description:** {sub_func_desc}
**Specification Reference:** {spec_ref}

---

**SUMMARIES (for context):**
{"".join(summary_text) if summary_text else "No summaries available"}

---

**ORIGINAL SPECIFICATION (for details):**
{"".join(original_text) if original_text else "No original content available"}

---

{feedback_text}

**Your Task:**
Create a comprehensive implementation description for "{sub_func_name}" that an RTL engineer 
can use to write code. Extract all necessary information from the specification above.

Return ONLY a JSON object (no markdown, no explanation) with this structure:
{{
  "sub_function": "{sub_func_name}",
  "description": "One clear sentence describing what this does",
  "inputs": ["input_name (data type and size)", "..."],
  "outputs": ["output_name (data type and size)", "..."],
  "functionality": "Detailed explanation of the algorithm/operation",
  "steps": [
    "1. First implementable step with specifics",
    "2. Second implementable step with specifics",
    "..."
  ],
  "references": ["Section X.Y.Z", "Figure N", "Table M"],
  "implementation_notes": "Critical details: data types, constraints, edge cases, etc."
}}

Be thorough and precise. Include all information needed for RTL implementation.
"""
        
        return prompt
    
    def _extract_section_id(self, spec_reference: str) -> Optional[str]:
        """Extract section ID from reference string."""
        if not spec_reference:
            return None
        
        # Handle formats like "Section 5.1.1" or "Sec. 5.1.1"
        import re
        match = re.search(r'(\d+\.[\d\.]+)', spec_reference)
        if match:
            return match.group(1)
        
        return None
    
    def _parse_description_response(self, response_text: str) -> Optional[Dict]:
        """
        Parse LLM response into structured description.
        
        Handles:
        - Pure JSON
        - JSON wrapped in markdown code blocks
        - Malformed responses
        """
        logger.debug("Parsing description response...")
        
        # Clean response
        text = response_text.strip()
        
        # Remove markdown code fences if present
        if text.startswith("```"):
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
            description = json.loads(text)
            logger.debug(f"✓ Parsed description for: {description.get('sub_function', 'Unknown')}")
            return description
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            logger.debug(f"Response text: {text[:500]}...")
            return None
    
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

async def test_description_agent():
    """Test Description Agent with all sub-functions."""
    import sys
    
    print("🚀 Testing Description Agent", flush=True)
    print("="*70, flush=True)
    
    try:
        from src.core.llm.openai_provider import OpenAIProvider
        from src.core.logging_config import setup_logging
        
        logger = setup_logging()
        logger.info("="*70)
        logger.info("STARTING DESCRIPTION AGENT TEST")
        logger.info("="*70)
        
        # 1. Load summaries
        print("\n📄 Loading summaries...", flush=True)
        summary_files = sorted(Path('data/output/summaries').glob('summaries_*.json'))
        
        if not summary_files:
            print("❌ No summaries found!", flush=True)
            return
        
        with open(summary_files[-1]) as f:
            summary_data = json.load(f)
        
        summaries = summary_data.get('summaries', [])
        print(f"   ✓ Loaded {len(summaries)} summaries", flush=True)
        
        # 2. Extract original sections from summaries
        print("\n📄 Extracting original content...", flush=True)
        original_sections = []
        for summary in summaries:
            original_sections.append({
                'section_id': summary['section_id'],
                'title': summary['title'],
                'content': summary.get('original_content', '')
            })
        print(f"   ✓ Extracted {len(original_sections)} sections", flush=True)
        
        # 3. Load decomposition
        print("\n📄 Loading decomposition...", flush=True)
        decomp_files = sorted(Path('data/output/decompositions').glob('decomposition_*.json'))
        
        if not decomp_files:
            print("❌ No decomposition found! Run Decomposer Agent first!", flush=True)
            return
        
        with open(decomp_files[-1]) as f:
            decomp_data = json.load(f)
        
        sub_functions = decomp_data['decomposition']['sub_functions']
        print(f"   ✓ Loaded {len(sub_functions)} sub-functions", flush=True)
        
        # 4. Create agent
        print("\n🤖 Creating Description Agent...", flush=True)
        provider = OpenAIProvider(
            model_name="gpt-4o-mini",
            temperature=0.3
        )
        
        model_client = provider.create_model_client()
        agent = DescriptionAgent(model_client)
        print("   ✓ Agent created", flush=True)
        
        # 5. Process ALL sub-functions
        print(f"\n🔨 Describing {len(sub_functions)} sub-functions...", flush=True)
        
        descriptions = []
        
        for i, sub_func in enumerate(sub_functions, 1):
            print(f"\n[{i}/{len(sub_functions)}] Processing: {sub_func['name']}", flush=True)
            print("   This may take 10-30 seconds...", flush=True)
            
            # Log prompt only for first one
            log_prompt = (i == 1)
            
            description = await agent.describe_sub_function(
                sub_function=sub_func,
                summaries=summaries,
                original_sections=original_sections,
                log_prompt=log_prompt
            )
            
            if description:
                descriptions.append(description)
                print(f"   ✓ Completed: {sub_func['name']}", flush=True)
            else:
                print(f"   ❌ Failed: {sub_func['name']}", flush=True)
        
        # 6. Display results
        print(f"\n✅ Processed {len(descriptions)}/{len(sub_functions)} sub-functions", flush=True)
        print("="*70, flush=True)
        
        for desc in descriptions:
            print(f"\n📋 {desc['sub_function']}", flush=True)
            print(f"   {desc['description']}", flush=True)
            print(f"   Inputs: {len(desc.get('inputs', []))}", flush=True)
            print(f"   Outputs: {len(desc.get('outputs', []))}", flush=True)
            print(f"   Steps: {len(desc.get('steps', []))}", flush=True)
        
        # 7. Save all descriptions
        output_dir = Path("data/output/descriptions")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save individual files
        for desc in descriptions:
            output_file = output_dir / f"description_{desc['sub_function']}.json"
            with open(output_file, 'w') as f:
                json.dump(desc, f, indent=2)
            print(f"   💾 Saved: {output_file.name}", flush=True)
        
        # Save combined file
        combined_file = output_dir / f"all_descriptions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        combined_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "num_descriptions": len(descriptions),
                "input_tokens": agent.total_input_tokens,
                "output_tokens": agent.total_output_tokens,
                "estimated_cost": agent.total_cost
            },
            "descriptions": descriptions
        }
        
        with open(combined_file, 'w') as f:
            json.dump(combined_data, f, indent=2)
        
        print(f"\n   💾 Combined: {combined_file.name}", flush=True)
        
        # Show usage
        usage = agent.get_usage_summary()
        print(f"\n💰 Total Cost: ${usage['estimated_cost']:.4f}", flush=True)
        print(f"   Input: {usage['input_tokens']:,} tokens", flush=True)
        print(f"   Output: {usage['output_tokens']:,} tokens", flush=True)
        
        print("\n" + "="*70, flush=True)
        print("✅ TEST COMPLETE", flush=True)
        print("="*70, flush=True)
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    print("Starting description_agent.py...", flush=True)
    asyncio.run(test_description_agent())
    print("Finished.", flush=True)