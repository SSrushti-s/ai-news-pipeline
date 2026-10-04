"""
Multi-Tier LLM Extraction Engine.

Fallback chain:

    Groq Primary
        ↓
    Gemini
        ↓
    Groq Secondary

IMPORTANT:

Both Groq tiers use the EXACT SAME:

    - Groq API keys
    - Groq models

If there are:

    5 Groq API keys
    4 Groq models

then each Groq tier has:

    5 × 4 = 20 possible combinations

Therefore the maximum possible Groq attempts across both
Groq tiers is:

    20 + 20 = 40

The second Groq tier is NOT a different set of keys.
It intentionally reuses the same configured Groq key/model pool.
"""

import asyncio
import json
import logging
import random
from abc import ABC, abstractmethod
from typing import Optional

from .settings import LLM
from .chunking import chunk_text, estimate_tokens


logger = logging.getLogger("graphone.llm")


# Required fields that every successful LLM response must contain.
REQUIRED_KEYS = {
    "aiSummary",
    "filterTags",
    "topics",
}


# ================================================================
# ERRORS
# ================================================================


class LLMTierError(Exception):
    """
    Represents a failure inside a particular LLM tier.
    """

    def __init__(
        self,
        tier: str,
        reason: str,
    ):
        self.tier = tier
        self.reason = reason

        super().__init__(
            f"{tier}: {reason}"
        )


# ================================================================
# BASE TIER
# ================================================================


class BaseLLMTier(ABC):

    name: str

    @abstractmethod
    async def call(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """
        Return raw text response.

        Raise LLMTierError on failure.
        """
        ...


# ================================================================
# GEMINI
# ================================================================


class GeminiTier(BaseLLMTier):

    name = "gemini"

    async def call(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:

        import aiohttp

        if not LLM.gemini_api_key:
            raise LLMTierError(
                self.name,
                "no_api_key",
            )

        url = (
            "https://generativelanguage.googleapis.com/"
            "v1beta/models/"
            f"{LLM.gemini_model}"
            f":generateContent?key={LLM.gemini_api_key}"
        )

        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": (
                                f"{system_prompt}\n\n"
                                f"{user_prompt}"
                            )
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": (
                    LLM.max_output_tokens
                ),
                "responseMimeType": (
                    "application/json"
                ),
            },
        }

        try:

            async with aiohttp.ClientSession() as session:

                async with session.post(
                    url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(
                        total=60
                    ),
                ) as resp:

                    if resp.status == 429:

                        raise LLMTierError(
                            self.name,
                            "rate_limited",
                        )

                    if resp.status == 413:

                        raise LLMTierError(
                            self.name,
                            "payload_too_large",
                        )

                    if resp.status >= 400:

                        body = await resp.text()

                        raise LLMTierError(
                            self.name,
                            (
                                f"http_{resp.status}: "
                                f"{body[:200]}"
                            ),
                        )

                    data = await resp.json()

                    try:

                        return (
                            data[
                                "candidates"
                            ][0][
                                "content"
                            ][
                                "parts"
                            ][0][
                                "text"
                            ]
                        )

                    except (
                        KeyError,
                        IndexError,
                    ) as exc:

                        raise LLMTierError(
                            self.name,
                            (
                                "unexpected_response_shape: "
                                f"{exc}"
                            ),
                        )

        except LLMTierError:
            raise

        except Exception as exc:

            raise LLMTierError(
                self.name,
                f"request_error: {exc}",
            )


# ================================================================
# GROQ
# ================================================================


class GroqTier(BaseLLMTier):
    """
    Groq tier.

    Every Groq tier uses the SAME configured:

        LLM.groq_api_keys
        LLM.groq_models

    Example:

        5 keys
        ×
        4 models
        =
        20 combinations

    The tier tries:

        Key 1 → Model 1
        Key 1 → Model 2
        Key 1 → Model 3
        Key 1 → Model 4

        Key 2 → Model 1
        ...

        Key 5 → Model 4
    """

    def __init__(
        self,
        name: str,
    ):
        self.name = name

    async def call(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:

        import aiohttp

        # --------------------------------------------------------
        # Get ALL configured Groq keys
        # --------------------------------------------------------

        configured_keys = [
            key
            for key in (
                LLM.groq_api_keys
                or [LLM.groq_api_key]
            )
            if key
        ]

        # --------------------------------------------------------
        # Get ALL configured Groq models
        # --------------------------------------------------------

        configured_models = [
            model
            for model in (
                LLM.groq_models
                or [LLM.groq_model]
            )
            if model
        ]

        if not configured_keys:

            raise LLMTierError(
                self.name,
                "no_api_key",
            )

        if not configured_models:

            raise LLMTierError(
                self.name,
                "no_model",
            )

        logger.info(
            "[%s] Starting Groq pool: "
            "%d API keys × %d models = %d combinations",
            self.name,
            len(configured_keys),
            len(configured_models),
            (
                len(configured_keys)
                * len(configured_models)
            ),
        )

        last_error: Optional[LLMTierError] = None

        # ========================================================
        # KEY → MODEL
        # ========================================================

        for key_index, api_key in enumerate(
            configured_keys,
            start=1,
        ):

            for model_index, model_name in enumerate(
                configured_models,
                start=1,
            ):

                logger.info(
                    "[%s] Trying "
                    "model=%s, key=%d/%d",
                    self.name,
                    model_name,
                    key_index,
                    len(configured_keys),
                )

                url = (
                    "https://api.groq.com/"
                    "openai/v1/chat/completions"
                )

                headers = {
                    "Authorization": (
                        f"Bearer {api_key}"
                    ),
                    "Content-Type": (
                        "application/json"
                    ),
                }

                payload = {
                    "model": model_name,
                    "messages": [
                        {
                            "role": "system",
                            "content": system_prompt,
                        },
                        {
                            "role": "user",
                            "content": user_prompt,
                        },
                    ],
                    "temperature": 0.1,
                    "max_tokens": (
                        LLM.max_output_tokens
                    ),
                    "response_format": {
                        "type": "json_object"
                    },
                }

                try:

                    async with aiohttp.ClientSession() as session:

                        async with session.post(
                            url,
                            json=payload,
                            headers=headers,
                            timeout=aiohttp.ClientTimeout(
                                total=60
                            ),
                        ) as resp:

                            # ------------------------------------------------
                            # RATE LIMIT
                            # ------------------------------------------------

                            if resp.status == 429:

                                logger.warning(
                                    "[%s] 429 rate limited: "
                                    "key=%d/%d, model=%s",
                                    self.name,
                                    key_index,
                                    len(configured_keys),
                                    model_name,
                                )

                                last_error = LLMTierError(
                                    self.name,
                                    "rate_limited",
                                )

                                # Try the next key/model combination.
                                continue

                            # ------------------------------------------------
                            # PAYLOAD TOO LARGE
                            # ------------------------------------------------

                            if resp.status == 413:

                                logger.warning(
                                    "[%s] 413 payload too large: "
                                    "model=%s",
                                    self.name,
                                    model_name,
                                )

                                last_error = LLMTierError(
                                    self.name,
                                    "payload_too_large",
                                )

                                continue

                            # ------------------------------------------------
                            # OTHER HTTP ERRORS
                            # ------------------------------------------------

                            if resp.status >= 400:

                                body = await resp.text()

                                logger.warning(
                                    "[%s] HTTP %d: "
                                    "key=%d/%d, model=%s",
                                    self.name,
                                    resp.status,
                                    key_index,
                                    len(configured_keys),
                                    model_name,
                                )

                                last_error = LLMTierError(
                                    self.name,
                                    (
                                        f"http_{resp.status}: "
                                        f"{body[:200]}"
                                    ),
                                )

                                continue

                            # ------------------------------------------------
                            # SUCCESS
                            # ------------------------------------------------

                            data = await resp.json()

                            try:

                                content = (
                                    data[
                                        "choices"
                                    ][0][
                                        "message"
                                    ][
                                        "content"
                                    ]
                                )

                                logger.info(
                                    "[%s] SUCCESS: "
                                    "model=%s, key=%d/%d",
                                    self.name,
                                    model_name,
                                    key_index,
                                    len(configured_keys),
                                )

                                return content

                            except (
                                KeyError,
                                IndexError,
                            ) as exc:

                                last_error = LLMTierError(
                                    self.name,
                                    (
                                        "unexpected_response_shape: "
                                        f"{exc}"
                                    ),
                                )

                                continue

                except LLMTierError:
                    raise

                except Exception as exc:

                    logger.warning(
                        "[%s] Request error: "
                        "key=%d/%d, model=%s: %s",
                        self.name,
                        key_index,
                        len(configured_keys),
                        model_name,
                        exc,
                    )

                    last_error = LLMTierError(
                        self.name,
                        f"request_error: {exc}",
                    )

                    continue

        # ========================================================
        # ALL GROQ COMBINATIONS FAILED
        # ========================================================

        if last_error is not None:

            raise last_error

        raise LLMTierError(
            self.name,
            "all_groq_combinations_failed",
        )


# ================================================================
# DEEPSEEK
# ================================================================
#
# Kept here for compatibility.
#
# It is NOT part of the new fallback chain:
#
#     Groq → Gemini → Groq
#
# ================================================================


class DeepSeekTier(BaseLLMTier):

    name = "deepseek"

    async def call(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:

        import aiohttp

        if not LLM.deepseek_api_key:

            raise LLMTierError(
                self.name,
                "no_api_key",
            )

        url = (
            "https://api.deepseek.com/"
            "chat/completions"
        )

        headers = {
            "Authorization": (
                f"Bearer {LLM.deepseek_api_key}"
            ),
            "Content-Type": (
                "application/json"
            ),
        }

        payload = {
            "model": LLM.deepseek_model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            "temperature": 0.1,
            "max_tokens": (
                LLM.max_output_tokens
            ),
            "response_format": {
                "type": "json_object"
            },
        }

        try:

            async with aiohttp.ClientSession() as session:

                async with session.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(
                        total=60
                    ),
                ) as resp:

                    if resp.status == 429:

                        raise LLMTierError(
                            self.name,
                            "rate_limited",
                        )

                    if resp.status == 413:

                        raise LLMTierError(
                            self.name,
                            "payload_too_large",
                        )

                    if resp.status >= 400:

                        body = await resp.text()

                        raise LLMTierError(
                            self.name,
                            (
                                f"http_{resp.status}: "
                                f"{body[:200]}"
                            ),
                        )

                    data = await resp.json()

                    try:

                        return (
                            data[
                                "choices"
                            ][0][
                                "message"
                            ][
                                "content"
                            ]
                        )

                    except (
                        KeyError,
                        IndexError,
                    ) as exc:

                        raise LLMTierError(
                            self.name,
                            (
                                "unexpected_response_shape: "
                                f"{exc}"
                            ),
                        )

        except LLMTierError:
            raise

        except Exception as exc:

            raise LLMTierError(
                self.name,
                f"request_error: {exc}",
            )


# ================================================================
# TIER REGISTRY
# ================================================================

TIER_REGISTRY = {
    "gemini": GeminiTier,
    "deepseek": DeepSeekTier,
}


# ================================================================
# ORCHESTRATOR
# ================================================================


class LLMOrchestrator:
    """
    Runs the fallback chain against a single extraction task.

    Default chain:

        groq_primary
            ↓
        gemini
            ↓
        groq_secondary

    Both Groq tiers use the SAME:

        - 5 API keys
        - 4 models

    Chunking is performed before the first LLM request.
    """

    def __init__(
        self,
        chain: Optional[list[str]] = None,
    ):

        chain = chain or LLM.fallback_chain

        self.chain: list[BaseLLMTier] = []

        for tier_name in chain:

            # ----------------------------------------------------
            # First Groq pass
            # ----------------------------------------------------

            if tier_name == "groq_primary":

                self.chain.append(
                    GroqTier(
                        name="groq_primary"
                    )
                )

            # ----------------------------------------------------
            # Gemini
            # ----------------------------------------------------

            elif tier_name == "gemini":

                self.chain.append(
                    GeminiTier()
                )

            # ----------------------------------------------------
            # Second Groq pass
            # ----------------------------------------------------

            elif tier_name == "groq_secondary":

                self.chain.append(
                    GroqTier(
                        name="groq_secondary"
                    )
                )

            # ----------------------------------------------------
            # Any other registered tier
            # ----------------------------------------------------

            elif tier_name in TIER_REGISTRY:

                self.chain.append(
                    TIER_REGISTRY[tier_name]()
                )

            else:

                logger.warning(
                    "Unknown LLM tier '%s' "
                    "ignored.",
                    tier_name,
                )

        logger.info(
            "LLM fallback chain initialized: %s",
            [
                tier.name
                for tier in self.chain
            ],
        )

    # ============================================================
    # EXTRACT
    # ============================================================

    async def extract(
        self,
        system_prompt: str,
        raw_text: str,
    ) -> Optional[dict]:

        # --------------------------------------------------------
        # Pre-flight chunking
        # --------------------------------------------------------

        chunks = chunk_text(
            raw_text,
            max_tokens=LLM.max_input_tokens,
            overlap_tokens=(
                LLM.chunk_overlap_tokens
            ),
        )

        if len(chunks) > 1:

            logger.info(
                "Input split into %d chunks "
                "(~%d tokens total)",
                len(chunks),
                estimate_tokens(raw_text),
            )

        # --------------------------------------------------------
        # For this extraction task, use the first chunk.
        # --------------------------------------------------------

        primary_chunk = chunks[0]

        # --------------------------------------------------------
        # FALLBACK CHAIN
        # --------------------------------------------------------

        for tier in self.chain:

            logger.info(
                "Trying LLM tier: %s",
                tier.name,
            )

            result = await self._call_with_retry(
                tier,
                system_prompt,
                primary_chunk,
            )

            if result is not None:

                logger.info(
                    "LLM tier succeeded: %s",
                    tier.name,
                )

                return result

            logger.warning(
                "LLM tier exhausted: %s",
                tier.name,
            )

        # --------------------------------------------------------
        # EVERYTHING FAILED
        # --------------------------------------------------------

        logger.error(
            "All LLM tiers exhausted "
            "without a valid structured response"
        )

        return None

    # ============================================================
    # RETRY
    # ============================================================

    async def _call_with_retry(
        self,
        tier: BaseLLMTier,
        system_prompt: str,
        user_prompt: str,
    ) -> Optional[dict]:

        for attempt in range(
            LLM.max_retries_per_tier
        ):

            try:

                raw = await tier.call(
                    system_prompt,
                    user_prompt,
                )

                parsed = self._safe_json_parse(
                    raw
                )

                # ------------------------------------------------
                # Validate structured response
                # ------------------------------------------------

                if (
                    parsed is not None
                    and isinstance(parsed, dict)
                    and REQUIRED_KEYS.issubset(
                        parsed
                    )
                ):

                    logger.info(
                        "[%s] extraction succeeded",
                        tier.name,
                    )

                    return parsed

                logger.warning(
                    "[%s] schema-invalid output "
                    "(keys=%s): %r",
                    tier.name,
                    (
                        list(parsed.keys())
                        if isinstance(
                            parsed,
                            dict,
                        )
                        else type(
                            parsed
                        ).__name__
                    ),
                    raw[:300],
                )

                # Malformed JSON/schema:
                # immediately fall through.
                return None

            except LLMTierError as exc:

                # ------------------------------------------------
                # Rate limit
                # ------------------------------------------------
                #
                # IMPORTANT:
                #
                # GroqTier itself already tries every
                # key × model combination.
                #
                # Therefore, when the entire Groq pool
                # is exhausted, returning None allows
                # the orchestrator to move to Gemini.
                #
                # ------------------------------------------------

                if exc.reason == "rate_limited":

                    wait = min(
                        LLM.max_backoff_s,
                        LLM.max_backoff_s
                        * (
                            2 ** attempt
                        ),
                    )

                    wait += random.uniform(
                        0,
                        wait * 0.3,
                    )

                    logger.warning(
                        "[%s] 429, "
                        "backoff %.1fs "
                        "(tier attempt %d/%d)",
                        tier.name,
                        wait,
                        attempt + 1,
                        LLM.max_retries_per_tier,
                    )

                    await asyncio.sleep(
                        wait
                    )

                    continue

                # ------------------------------------------------
                # Payload too large
                # ------------------------------------------------

                if (
                    exc.reason
                    == "payload_too_large"
                ):

                    logger.warning(
                        "[%s] 413 despite "
                        "pre-chunking. "
                        "Falling through.",
                        tier.name,
                    )

                    return None

                # ------------------------------------------------
                # Other failure
                # ------------------------------------------------

                logger.warning(
                    "[%s] failed: %s. "
                    "Falling through to next tier.",
                    tier.name,
                    exc.reason,
                )

                return None

        logger.warning(
            "[%s] exhausted retries. "
            "Falling through.",
            tier.name,
        )

        return None

    # ============================================================
    # JSON PARSER
    # ============================================================

    def _safe_json_parse(
        self,
        raw_text: str,
    ) -> Optional[dict]:
        """
        Safely parse a JSON response.

        Handles both:

            {"aiSummary": ...}

        and:

            ```json
            {"aiSummary": ...}
            ```
        """

        if not raw_text:
            return None

        if not isinstance(
            raw_text,
            str,
        ):
            return None

        try:

            cleaned_text = raw_text.strip()

            # ----------------------------------------------------
            # Remove markdown fences if present
            # ----------------------------------------------------

            if cleaned_text.startswith(
                "```"
            ):

                cleaned_text = (
                    cleaned_text.split(
                        "\n",
                        1,
                    )[-1]
                )

                if cleaned_text.endswith(
                    "```"
                ):

                    cleaned_text = (
                        cleaned_text.rsplit(
                            "```",
                            1,
                        )[0]
                    )

            # ----------------------------------------------------
            # Parse JSON
            # ----------------------------------------------------

            return json.loads(
                cleaned_text.strip()
            )

        except Exception as exc:

            logger.warning(
                "Failed parsing raw JSON "
                "string block: %s",
                exc,
            )

            return None
