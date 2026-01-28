"""
Understanding Module Pipeline - Orchestrates all 4 agents with AutoGen GroupChat.

Implements the complete Understanding Module flow:
1. Understanding Agent (solo) - Summarizes specification
2. Decomposer Agent (solo) - Breaks down into sub-functions  
3. Description + Verifier Loop (GroupChat) - Refines each sub-function description

From paper: "The Understanding Module consists of three stages: Understanding, 
Decomposition, and Information Augmentation (Description + Verification)."
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime

from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_agentchat.messages import TextMessage

from src.agents.understanding.understanding_agent import UnderstandingAgent
from src.agents.understanding.decomposer_agent import DecomposerAgent
from src.agents.understanding.description_agent import DescriptionAgent
from src.agents.understanding.verifier_agent import VerifierAgent
from src.core.document_loader import DocumentLoader

logger = logging.getLogger("spec2rtl.understanding_pipeline")


class UnderstandingPipeline:
    """
    Complete Understanding Module pipeline.
    
    Orchestrates 4 agents:
    - Understanding Agent: Summarizes specification sections
    - Decomposer Agent: Breaks target function into sub-functions
    - Description Agent: Creates detailed implementation info
    - Verifier Agent: Validates and provides feedback
    
    Uses AutoGen GroupChat for Description + Verifier refinement loop.
    """
    
    def __init__(self, model_client):
        """
        Initialize pipeline with all 4 agents.
        
        Args:
            model_client: AutoGen model client (from OpenAIProvider)
        """
        logger.info("Initializing Understanding Pipeline")
        
        # Create all 4 agents
        self.understanding_agent = UnderstandingAgent(model_client)
        self.decomposer_agent = DecomposerAgent(model_client)
        self.description_agent = DescriptionAgent(model_client)
        self.verifier_agent = VerifierAgent(model_client)
        
        # Track overall usage
        self.total_cost = 0.0
        
        logger.info("✓ All agents initialized")
    
    async def run(
        self,
        spec_pdf_path: str,
        target_function: str,
        target_section_id: str,
        max_refinement_iterations: int = 3
    ) -> Dict:
        """
        Run complete Understanding Module pipeline.
        
        Args:
            spec_pdf_path: Path to specification PDF
            target_function: Name of function to implement (e.g., "AES Cipher")
            target_section_id: Section to focus on (e.g., "5.1")
            max_refinement_iterations: Max iterations for Description + Verifier loop
            
        Returns:
            Complete understanding output dict with summaries, decomposition, descriptions
        """
        logger.info("="*70)
        logger.info("STARTING UNDERSTANDING MODULE PIPELINE")
        logger.info("="*70)
        logger.info(f"Target: {target_function} (Section {target_section_id})")
        
        start_time = datetime.now()
        
        # ==================== STEP 1: UNDERSTANDING AGENT ====================
        print("\n" + "="*70)
        print("📖 STEP 1: Understanding Agent")
        print("="*70)
        logger.info("Step 1: Running Understanding Agent")

        # Load document
        loader = DocumentLoader(spec_pdf_path)
        pdf_path = loader.load()
        sections = loader.get_sections()  # ← FIXED

        print(f"📄 Loaded {len(sections)} sections from spec")

        # Process document
        summaries = await self.understanding_agent.process_document(sections)

        print(f"✅ Generated {len(summaries)} summaries")
        logger.info(f"Understanding Agent complete: {len(summaries)} summaries")
        logger.info(f"Understanding Agent complete: {len(summaries)} summaries")
        
        # ==================== STEP 2: DECOMPOSER AGENT ====================
        print("\n" + "="*70)
        print("🔨 STEP 2: Decomposer Agent")
        print("="*70)
        logger.info("Step 2: Running Decomposer Agent")
        
        # Extract original sections from summaries
        original_sections = []
        for summary in summaries:
            original_sections.append({
                'section_id': summary['section_id'],
                'title': summary['title'],
                'content': summary.get('original_content', '')
            })
        
        # Decompose target function
        decomposition = await self.decomposer_agent.decompose_function(
            target_function=target_function,
            summaries=summaries,
            original_sections=original_sections,
            target_section_id=target_section_id
        )
        
        sub_functions = decomposition['sub_functions']
        print(f"✅ Decomposed into {len(sub_functions)} sub-functions:")
        for sf in sub_functions:
            print(f"   {sf['order']}. {sf['name']}")
        
        logger.info(f"Decomposer Agent complete: {len(sub_functions)} sub-functions")
        
        # ==================== STEP 3: DESCRIPTION + VERIFIER LOOP ====================
        print("\n" + "="*70)
        print("📋 STEP 3: Description + Verifier Loop (AutoGen GroupChat)")
        print("="*70)
        logger.info("Step 3: Running Description + Verifier refinement")
        
        verified_descriptions = []
        
        for i, sub_func in enumerate(sub_functions, 1):
            print(f"\n[{i}/{len(sub_functions)}] Processing: {sub_func['name']}")
            logger.info(f"Processing sub-function {i}/{len(sub_functions)}: {sub_func['name']}")
            
            # Refine this sub-function with GroupChat
            verified_desc = await self._refine_with_groupchat(
                sub_func=sub_func,
                summaries=summaries,
                original_sections=original_sections,
                max_iterations=max_refinement_iterations
            )
            
            verified_descriptions.append(verified_desc)
            
            if verified_desc.get('verification', {}).get('is_valid'):
                print(f"   ✅ VERIFIED: {sub_func['name']}")
            else:
                print(f"   ⚠️  Max iterations reached: {sub_func['name']}")
        
        logger.info(f"Description + Verifier complete: {len(verified_descriptions)} verified")
        
        # ==================== FINALIZE ====================
        elapsed = (datetime.now() - start_time).total_seconds()
        
        # Calculate total cost
        self.total_cost = (
            self.understanding_agent.total_cost +
            self.decomposer_agent.total_cost +
            self.description_agent.total_cost +
            self.verifier_agent.total_cost
        )

        # ==================== GENERATE CLEAN IMPLEMENTATION PLAN ====================
        logger.info("Generating final implementation plan")
        print("\n📄 Generating final implementation plan...")

        implementation_plan = self._generate_implementation_plan(
            target_function=target_function,
            decomposition=decomposition,
            verified_descriptions=verified_descriptions
        )

        # Save implementation plan separately
        plan_file = self._save_implementation_plan(implementation_plan)
        
        # Build output
        output = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "target_function": target_function,
                "target_section": target_section_id,
                "num_summaries": len(summaries),
                "num_sub_functions": len(sub_functions),
                "num_descriptions": len(verified_descriptions),
                "elapsed_seconds": elapsed,
                "total_cost": self.total_cost
            },
            "summaries": summaries,
            "decomposition": decomposition,
            "descriptions": verified_descriptions
        }

        # Add to output
        output['implementation_plan'] = implementation_plan
        
        # Save complete output
        self._save_pipeline_output(output)
        
        print("\n" + "="*70)
        print("✅ UNDERSTANDING MODULE COMPLETE")
        print("="*70)
        print(f"📊 Summaries: {len(summaries)}")
        print(f"🔨 Sub-functions: {len(sub_functions)}")
        print(f"📋 Verified Descriptions: {len(verified_descriptions)}")
        print(f"⏱️  Time: {elapsed:.1f}s")
        print(f"💰 Total Cost: ${self.total_cost:.4f}")
        print("="*70)
        
        logger.info("="*70)
        logger.info("UNDERSTANDING MODULE PIPELINE COMPLETE")
        logger.info("="*70)
        
        return output
    
    async def _refine_with_groupchat(
        self,
        sub_func: Dict,
        summaries: List[Dict],
        original_sections: List[Dict],
        max_iterations: int = 3
    ) -> Dict:
        """
        Refine a sub-function description using AutoGen GroupChat.
        
        Description Agent and Verifier Agent work together in a conversation
        until the description is valid or max iterations reached.
        
        Args:
            sub_func: Sub-function dict from Decomposer
            summaries: Summaries from Understanding Agent
            original_sections: Original document sections
            max_iterations: Maximum refinement iterations
            
        Returns:
            Verified description dict
        """
        sub_func_name = sub_func['name']
        logger.info(f"Starting GroupChat refinement for: {sub_func_name}")
        
        # Initial description
        print(f"   💭 Generating initial description...")
        description = await self.description_agent.describe_sub_function(
            sub_function=sub_func,
            summaries=summaries,
            original_sections=original_sections
        )
        
        if not description:
            logger.error(f"Failed to generate description for {sub_func_name}")
            return None
        
        # Initial verification
        print(f"   🔍 Verifying...")
        verification = await self.verifier_agent.verify_description(description)
        
        if not verification:
            logger.error(f"Failed to verify description for {sub_func_name}")
            return None
        
        iteration = 1
        
        # Refinement loop
        while not verification.get('is_valid', False) and iteration < max_iterations:
            print(f"   🔄 Iteration {iteration + 1}: Refining based on feedback...")
            logger.info(f"Refinement iteration {iteration + 1} for {sub_func_name}")
            
            # Extract feedback
            issues = verification.get('issues', [])
            suggestions = verification.get('suggestions', [])
            feedback = issues + suggestions
            
            logger.debug(f"Feedback items: {len(feedback)}")
            
            # Revise description with feedback
            description = await self.description_agent.describe_sub_function(
                sub_function=sub_func,
                summaries=summaries,
                original_sections=original_sections,
                verifier_feedback=feedback
            )
            
            if not description:
                logger.error(f"Failed to revise description for {sub_func_name}")
                break
            
            # Re-verify
            verification = await self.verifier_agent.verify_description(description)
            
            if not verification:
                logger.error(f"Failed to re-verify description for {sub_func_name}")
                break
            
            iteration += 1
        
        # Final status
        if verification.get('is_valid', False):
            logger.info(f"✓ Description validated for {sub_func_name} (iteration {iteration})")
        else:
            logger.warning(f"✗ Max iterations reached for {sub_func_name} (not fully validated)")
        
        # Attach verification to description
        description['verification'] = verification
        description['refinement_iterations'] = iteration
        
        return description
    
    def _generate_implementation_plan(
        self,
        target_function: str,
        decomposition: Dict,
        verified_descriptions: List[Dict]
    ) -> Dict:
        """
        Generate clean implementation plan (like paper's Figure 4).
        
        This is the final, coding-ready output that removes all metadata,
        verification details, and refinement info. Only keeps what's needed
        for RTL implementation.
        """
        logger.info("Generating final implementation plan")
        
        # Build clean sub-function list
        implementation_plan = {
            "target_function": target_function,
            "implementation_plan": []
        }
        
        for desc in verified_descriptions:
            # Extract only essential fields for coding
            clean_desc = {
                "name": desc['sub_function'],
                "description": desc['description'],
                "inputs": desc.get('inputs', []),
                "outputs": desc.get('outputs', []),
                "functionality": desc.get('functionality', ''),
                "steps": desc.get('steps', []),
                "references": desc.get('references', []),
                "implementation_notes": desc.get('implementation_notes', '')
            }
            
            # Add order and dependencies from decomposition
            for sub_func in decomposition['sub_functions']:
                if sub_func['name'] == desc['sub_function']:
                    clean_desc['order'] = sub_func['order']
                    clean_desc['dependencies'] = sub_func.get('dependencies', [])
                    break
            
            implementation_plan['implementation_plan'].append(clean_desc)
        
        # Sort by order
        implementation_plan['implementation_plan'].sort(key=lambda x: x.get('order', 999))
        
        logger.info(f"Generated clean implementation plan with {len(implementation_plan['implementation_plan'])} sub-functions")
        
        return implementation_plan

    def _save_implementation_plan(self, plan: Dict):
        """Save clean implementation plan (coding-ready format)."""
        output_dir = Path("data/output/implementation_plans")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        target_name = plan['target_function'].replace(' ', '_')
        output_file = output_dir / f"implementation_plan_{target_name}_{timestamp}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(plan, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Implementation plan saved to: {output_file}")
        print(f"📄 Implementation Plan saved to: {output_file}")
        
        return output_file
        
    def _save_pipeline_output(self, output: Dict):
        """Save complete pipeline output to file."""
        output_dir = Path("data/output/understanding_module")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"understanding_complete_{timestamp}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Pipeline output saved to: {output_file}")
        print(f"\n💾 Saved to: {output_file}")
    
    def get_usage_summary(self) -> Dict:
        """Get aggregated usage statistics from all agents."""
        return {
            "understanding_agent": self.understanding_agent.get_usage_summary(),
            "decomposer_agent": self.decomposer_agent.get_usage_summary(),
            "description_agent": self.description_agent.get_usage_summary(),
            "verifier_agent": self.verifier_agent.get_usage_summary(),
            "total_cost": self.total_cost
        }


async def test_understanding_pipeline():
    """Test complete Understanding Pipeline."""
    from src.core.llm.openai_provider import OpenAIProvider
    from src.core.logging_config import setup_logging
    
    logger = setup_logging()
    
    print("🚀 Testing Complete Understanding Pipeline", flush=True)
    print("="*70, flush=True)
    
    try:
        # Create model client
        provider = OpenAIProvider(
            model_name="gpt-4o-mini",
            temperature=0.3
        )
        model_client = provider.create_model_client()
        
        # Create pipeline
        pipeline = UnderstandingPipeline(model_client)
        
        # Run complete pipeline
        result = await pipeline.run(
            spec_pdf_path="data/input/specs/riscv-spec-20191213.pdf",
            target_function="RISC V 32I",
            target_section_id="2",
            max_refinement_iterations=3
        )
        
        # Display usage breakdown
        print("\n" + "="*70)
        print("💰 COST BREAKDOWN")
        print("="*70)
        usage = pipeline.get_usage_summary()
        
        for agent_name, agent_usage in usage.items():
            if agent_name == "total_cost":
                continue
            print(f"\n{agent_name}:")
            print(f"  Input:  {agent_usage['input_tokens']:,} tokens")
            print(f"  Output: {agent_usage['output_tokens']:,} tokens")
            print(f"  Cost:   ${agent_usage['estimated_cost']:.4f}")
        
        print(f"\n{'─'*70}")
        print(f"TOTAL COST: ${usage['total_cost']:.4f}")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    print("Starting understanding_pipeline.py...", flush=True)
    # asyncio.run(test_understanding_pipeline())
    asyncio.run(test_understanding_pipeline())
    print("Finished.", flush=True)