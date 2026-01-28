"""
Verifier Agent - Validates and provides feedback on descriptions.

From paper: "The Verifier then reviews this dictionary, providing feedback 
to enhance the quality and accuracy of the information."
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage

logger = logging.getLogger("spec2rtl.verifier_agent")


class VerifierAgent:
    """
    Fourth agent in Understanding Module.
    Verifies and provides feedback on sub-function descriptions.
    """
    
    SYSTEM_PROMPT = """You are an expert RTL verification specialist.

Your task: Verify the completeness and correctness of hardware sub-function descriptions.

For each description, you must check:

1. **Completeness:**
   - All required fields present (sub_function, description, inputs, outputs, functionality, steps, references, implementation_notes)
   - Inputs/outputs have data types and sizes
   - Steps are numbered and complete
   - References are specific (not vague)

2. **Correctness:**
   - Description accurately reflects the specification
   - Inputs/outputs match the specification
   - Steps are logically ordered and implementable
   - References point to correct sections

3. **Clarity:**
   - Steps are clear and unambiguous
   - Technical terminology is precise
   - Edge cases are addressed
   - Implementation notes are helpful

4. **RTL Readiness:**
   - Information is sufficient for RTL coding
   - Data types are specified
   - Algorithm is concrete (not abstract)
   - No missing critical details

Output format: JSON with structure:
{
  "is_valid": true or false,
  "confidence": 0.0 to 1.0,
  "feedback": [
    "✓ Positive feedback item",
    "✓ Another good point"
  ],
  "issues": [
    "Missing: specific issue",
    "Unclear: specific problem"
  ],
  "suggestions": [
    "Add: specific improvement",
    "Clarify: specific detail"
  ]
}

Be thorough and constructive. Your feedback will be used to improve the description."""
    
    def __init__(self, model_client):
        """
        Initialize Verifier Agent.
        
        Args:
            model_client: AutoGen model client (from OpenAIProvider)
        """
        self.agent = AssistantAgent(
            name="verifier_agent",
            model_client=model_client,
            system_message=self.SYSTEM_PROMPT
        )
        
        # Track usage
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
    
    async def verify_description(
        self,
        description: Dict,
        log_prompt: bool = False
    ) -> Dict:
        """
        Verify a sub-function description.
        
        Args:
            description: Description dict from Description Agent
            log_prompt: If True, log the full prompt
            
        Returns:
            Verification result dict
        """
        sub_func_name = description.get('sub_function', 'Unknown')
        
        logger.info(f"Verifying description: {sub_func_name}")
        
        # Build prompt
        prompt = self._build_verification_prompt(description)
        
        if log_prompt:
            logger.debug("="*70)
            logger.debug("VERIFIER PROMPT:")
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
        verification = self._parse_verification_response(response_text)
        
        # Validate
        if not verification:
            logger.error("Failed to parse valid verification")
            return None
        
        is_valid = verification.get('is_valid', False)
        confidence = verification.get('confidence', 0.0)
        
        if is_valid:
            logger.info(f"✓ VALID (confidence: {confidence:.2f}): {sub_func_name}")
        else:
            num_issues = len(verification.get('issues', []))
            logger.warning(f"✗ INVALID ({num_issues} issues): {sub_func_name}")
        
        return verification
    
    def _build_verification_prompt(self, description: Dict) -> str:
        """Build the verification prompt."""
        
        # Convert description to formatted string
        desc_json = json.dumps(description, indent=2)
        
        prompt = f"""Verify the following sub-function description for completeness, correctness, and RTL readiness:
```json
{desc_json}
```

**Your Task:**
Carefully review the description above and check:

1. **Completeness** - Are all required fields present and filled?
   - sub_function, description, inputs, outputs, functionality, steps, references, implementation_notes
   - Do inputs/outputs specify data types and sizes?
   - Are steps numbered and complete?
   - Are references specific?

2. **Correctness** - Does the description accurately reflect what's needed?
   - Are the inputs/outputs correct for this operation?
   - Are the steps logically ordered?
   - Are the references valid?

3. **Clarity** - Is everything clear and unambiguous?
   - Can an RTL engineer understand and implement this?
   - Are edge cases addressed?
   - Is terminology precise?

4. **RTL Readiness** - Is this ready for coding?
   - Are data types specified?
   - Is the algorithm concrete?
   - Are there any missing critical details?

Return ONLY a JSON object (no markdown, no explanation) with this structure:
{{
  "is_valid": true or false,
  "confidence": 0.0 to 1.0,
  "feedback": [
    "✓ Specific positive feedback",
    "✓ Another strength"
  ],
  "issues": [
    "Missing: specific issue",
    "Unclear: specific problem"
  ],
  "suggestions": [
    "Add: specific improvement",
    "Clarify: specific detail needed"
  ]
}}

If is_valid=false, provide specific, actionable feedback in "issues" and "suggestions".
If is_valid=true, still provide constructive feedback and suggestions for improvement.
"""
        
        return prompt
    
    def _parse_verification_response(self, response_text: str) -> Optional[Dict]:
        """
        Parse LLM response into structured verification.
        
        Handles:
        - Pure JSON
        - JSON wrapped in markdown code blocks
        - Malformed responses
        """
        logger.debug("Parsing verification response...")
        
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
            verification = json.loads(text)
            is_valid = verification.get('is_valid', False)
            logger.debug(f"✓ Parsed verification: valid={is_valid}")
            return verification
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


async def test_verifier_agent():
    """Test Verifier Agent with a description."""
    import sys
    
    print("🚀 Testing Verifier Agent", flush=True)
    print("="*70, flush=True)
    
    try:
        from src.core.llm.openai_provider import OpenAIProvider
        from src.core.logging_config import setup_logging
        
        logger = setup_logging()
        logger.info("="*70)
        logger.info("STARTING VERIFIER AGENT TEST")
        logger.info("="*70)
        
        # 1. Load descriptions
        print("\n📄 Loading descriptions...", flush=True)
        desc_files = sorted(Path('data/output/descriptions').glob('all_descriptions_*.json'))
        
        if not desc_files:
            print("❌ No descriptions found! Run Description Agent first!", flush=True)
            return
        
        with open(desc_files[-1]) as f:
            desc_data = json.load(f)
        
        descriptions = desc_data.get('descriptions', [])
        print(f"   ✓ Loaded {len(descriptions)} descriptions", flush=True)
        
        # 2. Create agent
        print("\n🤖 Creating Verifier Agent...", flush=True)
        provider = OpenAIProvider(
            model_name="gpt-5.2",
            temperature=0.3
        )
        
        model_client = provider.create_model_client()
        agent = VerifierAgent(model_client)
        print("   ✓ Agent created", flush=True)
        
        # 3. Test with first description
        test_desc = descriptions[0]
        print(f"\n🔍 Verifying: {test_desc['sub_function']}", flush=True)
        print("   This may take 10-30 seconds...", flush=True)
        
        verification = await agent.verify_description(
            description=test_desc,
            log_prompt=True
        )
        
        if verification:
            print("\n✅ Verification Complete", flush=True)
            print("="*70, flush=True)
            print(f"Sub-function: {test_desc['sub_function']}", flush=True)
            print(f"Valid: {verification['is_valid']}", flush=True)
            print(f"Confidence: {verification.get('confidence', 0.0):.2f}", flush=True)
            
            print(f"\n✓ Feedback ({len(verification.get('feedback', []))}):", flush=True)
            for fb in verification.get('feedback', []):
                print(f"  {fb}", flush=True)
            
            if verification.get('issues'):
                print(f"\n✗ Issues ({len(verification['issues'])}):", flush=True)
                for issue in verification['issues']:
                    print(f"  {issue}", flush=True)
            
            if verification.get('suggestions'):
                print(f"\n💡 Suggestions ({len(verification['suggestions'])}):", flush=True)
                for sug in verification['suggestions']:
                    print(f"  {sug}", flush=True)
            
            # Save
            output_dir = Path("data/output/verifications")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f"verification_{test_desc['sub_function']}.json"
            
            with open(output_file, 'w') as f:
                json.dump({
                    "description": test_desc,
                    "verification": verification
                }, f, indent=2)
            
            print(f"\n💾 Saved to: {output_file}", flush=True)
            
            # Show usage
            usage = agent.get_usage_summary()
            print(f"\n💰 Cost: ${usage['estimated_cost']:.4f}", flush=True)
            print(f"   Input: {usage['input_tokens']:,} tokens", flush=True)
            print(f"   Output: {usage['output_tokens']:,} tokens", flush=True)
            
            print("\n" + "="*70, flush=True)
            print("✅ TEST COMPLETE", flush=True)
        else:
            print("\n❌ Verification failed", flush=True)
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    print("Starting verifier_agent.py...", flush=True)
    asyncio.run(test_verifier_agent())
    print("Finished.", flush=True)