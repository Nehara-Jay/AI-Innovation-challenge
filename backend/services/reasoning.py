"""
Reasoning Service (Person 1)
Handles OpenRouter / DeepSeek-R1 LLM synthesis, query decomposition,
conflict resolution between documents and ephemera, and citation generation.
"""

import re
import json
from typing import List, Tuple, Optional, Dict, Any
from openai import OpenAI

from backend.config import settings
from backend.models import SearchHit, ReasoningStep


class ReasoningService:
    def __init__(self):
        self._client: Optional[OpenAI] = None

    def _get_client(self) -> OpenAI:
        """Lazily initializes and returns the OpenRouter client."""
        if self._client is None:
            if not settings.openrouter_api_key:
                raise ValueError("OPENROUTER_API_KEY is not configured in .env")
            self._client = OpenAI(
                base_url=settings.openrouter_base_url,
                api_key=settings.openrouter_api_key,
            )
        return self._client

    def analyze_image(self, image_path: str, question: str) -> str:
        """
        Uses Gemini 2.5 Flash Vision via OpenRouter to analyze and describe image plates.
        """
        import base64
        from pathlib import Path

        p = Path(image_path)
        if not p.exists():
            return ""

        try:
            with open(p, "rb") as f:
                b64_data = base64.b64encode(f.read()).decode("utf-8")

            client = self._get_client()
            resp = client.chat.completions.create(
                model="google/gemini-2.5-flash",
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Analyze this fantasy lore image/plate and answer the following question accurately: {question}",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64_data}"},
                        },
                    ],
                }],
                max_tokens=512,
                temperature=0.1,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"[Vision Error] Could not analyze image {p.name}: {e}")
            return ""

    def decompose_query(self, question: str) -> List[str]:
        """
        Decomposes complex multi-hop questions into 1 or 2 targeted search queries.
        Example:
          Input: 'What practice was outlawed by the ruler who commissioned the Sunken Relic?'
          Output: ['Who commissioned the Sunken Relic?', 'What practice did the ruler ban or outlaw?']
        """
        client = self._get_client()
        prompt = f"""You are a search query planner for a multi-hop document retrieval system.
Analyze the user's question and break it down into 1 or 2 clear, standalone search sub-queries to find the answer across multiple documents.

If the question is already simple and single-hop, return only 1 search query.
If the question requires connecting facts across multiple files, return 2 sequential search queries.

User Question: "{question}"

Respond with ONLY a valid JSON list of strings, for example:
["first search query", "second search query"]
"""

        try:
            response = client.chat.completions.create(
                model=settings.llm_model,
                messages=[
                    {"role": "system", "content": "You are a concise search query decomposition engine. Output only valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=256,
            )
            raw_text = response.choices[0].message.content.strip()

            # Clean any DeepSeek <think>...</think> reasoning tags or markdown blocks
            raw_text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
            if raw_text.startswith("```"):
                raw_text = re.sub(r"^```[a-z]*\n", "", raw_text)
                raw_text = re.sub(r"\n```$", "", raw_text)

            parsed = json.loads(raw_text)
            if isinstance(parsed, list) and len(parsed) > 0:
                return [str(q).strip() for q in parsed if str(q).strip()]
        except Exception as e:
            # Fallback to original question if decomposition fails
            print(f"[Decomposition Notice] Falling back to original query: {e}")

        return [question]

    def synthesize_answer(
        self,
        question: str,
        context_chunks: List[SearchHit],
        reasoning_steps: List[ReasoningStep],
    ) -> Tuple[str, float]:
        """
        Generates an evidence-backed answer using DeepSeek-R1 / OpenRouter with citations.
        Returns: (synthesized_answer, estimated_confidence)
        """
        if not context_chunks:
            return (
                "I could not find any relevant information in the archive to answer this question.",
                0.0,
            )

        client = self._get_client()

        # Format context passages with citation indices [1], [2]...
        formatted_context = []
        for i, chunk in enumerate(context_chunks):
            passage_type = "Official Document / Codex" if chunk.type == "document" else "Ephemera / In-world Account (Unreliable Narrator)"
            formatted_context.append(
                f"[{i + 1}] Source: {chunk.source} (Page {chunk.page}) | Type: {passage_type}\n"
                f"Content: {chunk.content.strip()}\n"
            )
        context_text = "\n".join(formatted_context)

        system_prompt = """You are an expert lore archivist and reasoning assistant for the "Ashen Era Archive".
Your goal is to answer questions accurately and strictly based on the provided document excerpts.

CRITICAL GUIDELINES:
1. Answer strictly from the provided context. Do NOT invent facts or hallucinate outside lore.
2. Provide citations using bracketed numbers [1], [2], etc., matching the provided sources.
3. CONFLICT RESOLUTION: If an ephemera/letter contradicts an official codex or chronicle, state the official fact first and explicitly mention the conflicting rumor/account from the ephemera.
4. If facts are connected across multiple sources (e.g., entity mentioned in [1] and action in [2]), clearly explain the connection.
5. If the context does not contain enough information to fully answer the question, state what is known and mention what information is missing.
"""

        user_prompt = f"""Context Passages:
----------------------------------------
{context_text}
----------------------------------------

Question: {question}

Provide a comprehensive, evidence-backed answer with citations:"""

        try:
            response = client.chat.completions.create(
                model=settings.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=settings.temperature,
                max_tokens=settings.max_tokens,
            )

            message = response.choices[0].message
            raw_answer = message.content or getattr(message, "reasoning", "") or ""
            raw_answer = raw_answer.strip()

            # Clean think tags if model emitted CoT tags
            if "<think>" in raw_answer and "</think>" in raw_answer:
                think_match = re.search(r"<think>(.*?)</think>", raw_answer, flags=re.DOTALL)
                if think_match:
                    thought_content = think_match.group(1).strip()
                    if thought_content and reasoning_steps:
                        reasoning_steps[-1].thought = thought_content[:500] + "..." if len(thought_content) > 500 else thought_content
                raw_answer = re.sub(r"<think>.*?</think>", "", raw_answer, flags=re.DOTALL).strip()

            # If reasoning attribute was present separately
            reasoning_attr = getattr(message, "reasoning", None)
            if reasoning_attr and reasoning_steps:
                reasoning_steps[-1].thought = str(reasoning_attr)[:500] + "..."

            confidence = 0.90 if len(context_chunks) >= 2 else 0.75

            return raw_answer, confidence

        except Exception as e:
            return f"An error occurred while synthesizing the answer: {e}", 0.0


# Singleton instance
reasoning_service = ReasoningService()
