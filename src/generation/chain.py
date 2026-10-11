"""Answer generation: prompt, LLM, and context formatting.

Kept separate from retrieval so the two can be evaluated independently --
a bad answer is either a retrieval failure (wrong chunks) or a generation
failure (right chunks, wrong answer), and conflating them makes both harder
to fix.
"""

from __future__ import annotations

import os

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

DEFAULT_MODEL = "llama-3.1-8b-instant"

# temperature=0.2 rather than 0: some variation reads better, but grounded
# answers should not be creative.
DEFAULT_TEMPERATURE = 0.2

ANSWER_PROMPT = PromptTemplate.from_template("""
You are an expert research assistant. Use the context below to give a detailed, thorough answer.
- Write at least 3-5 sentences
- Use bullet points for lists or multiple points
- Always cite which source each piece of information came from
- If multiple sources mention the topic, combine their information
- Only use information from the context
- If the context does not contain the answer, say so plainly rather than guessing

Context:
{context}

Question: {question}

Detailed Answer:""")


def format_docs(documents: list[Document]) -> str:
    """Render retrieved chunks as labelled context.

    The [Source: ...] prefix is what lets the model attribute claims, and what
    makes the citations in the UI verifiable rather than decorative.
    """
    return "\n\n".join(
        f"[Source: {document.metadata.get('source', 'unknown')}]\n{document.page_content}"
        for document in documents
    )


def build_chain(
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    api_key: str | None = None,
):
    """Build the generation chain: prompt -> LLM -> string.

    Raises:
        ValueError: If no Groq API key is available.
    """
    key = api_key or os.getenv("GROQ_API_KEY")
    if not key:
        raise ValueError(
            "GROQ_API_KEY is not set. Create a .env file containing:\n"
            "  GROQ_API_KEY=your_key_here\n"
            "Get a free key at https://console.groq.com"
        )

    llm = ChatGroq(model=model, api_key=key, temperature=temperature)
    return ANSWER_PROMPT | llm | StrOutputParser()
