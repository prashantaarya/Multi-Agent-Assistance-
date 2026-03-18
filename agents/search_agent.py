# agents/search_agent.py
import aiohttp
import asyncio
import urllib.parse
import logging
import warnings
from typing import List, Tuple, Optional

# Suppress RuntimeWarning from the old duckduckgo_search package
# (it was renamed to ddgs; both may be installed during transition)
warnings.filterwarnings(
    "ignore",
    message=r".*duckduckgo.search.*renamed.*ddgs.*",
    category=RuntimeWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*ddgs.*renamed.*",
    category=RuntimeWarning,
)

from autogen_agentchat.agents import AssistantAgent
from core.capabilities import register
from core.schemas import ToolParameter, ParameterType

# Real web search via ddgs (formerly duckduckgo_search)
try:
    from ddgs import DDGS
    _DDGS_AVAILABLE = True
except ImportError:
    try:
        from duckduckgo_search import DDGS
        _DDGS_AVAILABLE = True
    except ImportError:
        _DDGS_AVAILABLE = False

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

# ---- Feature toggles ---------------------------------------------------------
# Keep OFF by default (saves tokens). Turn ON to improve recall on very noisy queries.
ENABLE_LLM_REWRITE = False

# Words/phrases to strip from the user query before searching
_STOP_TERMS = {
    "brief", "brif", "overview", "in brief", "summary", "explain", "about",
    "info", "information", "tell me", "give", "give me", "provide", "please",
    "can you", "could you", "would you", "help me", "on", "of"
}


class SearchAgent(AssistantAgent):
    """
    Factual web lookups using DuckDuckGo + Wikipedia with robust title resolution.
    One capability exposed to the orchestrator:  search.web

    Optional inputs (from planner/orchestrator):
      - query:  str  (required)
      - source: "auto" | "wiki" | "ddg" (default "auto")
    """

    def __init__(self, name: str = "search", model_client=None):
        super().__init__(
            name=name,
            model_client=model_client,
            system_message=(
                "You are the SearchAgent. Given a query, fetch factual information from Wikipedia "
                "and DuckDuckGo Instant Answer, then produce a concise, accurate summary using the LLM. "
                "Avoid speculation. Prefer verifiable facts. Include a short 'Sources' section when available."
            ),
        )
        # Keep stable references
        self._llm_client = model_client
        # Small internal agent used for LLM rewrites/summaries via group chat
        self._summarizer = AssistantAgent(
            name="search_summarizer",
            model_client=model_client,
            system_message="You write concise, factual summaries based strictly on provided source text."
        )

        # ──────────────────────────────────────────────────────────────────────
        # Register capability with FULL SCHEMA (Industry Best Practice)
        # ──────────────────────────────────────────────────────────────────────
        
        register(
            capability="search.web",
            agent_name=self.name,
            handler=self.search,
            description="Search the web for factual information using Wikipedia and DuckDuckGo. Best for who/what/when/where/why/how questions, historical facts, definitions, and current events.",
            parameters=[
                ToolParameter(
                    name="query",
                    type=ParameterType.STRING,
                    description="The search query or question to look up",
                    required=True
                ),
                ToolParameter(
                    name="source",
                    type=ParameterType.STRING,
                    description="The search source to use",
                    required=False,
                    default="auto",
                    enum=["auto", "wiki", "ddg"]
                )
            ],
            category="search",
            examples=[
                {"query": "Who invented Python programming language"},
                {"query": "When did World War 2 end"},
                {"query": "What is the capital of France", "source": "wiki"}
            ]
        )

    # -------------------------------------------------------------------------
    # Public capability handler
    # -------------------------------------------------------------------------
    async def search(self, query: str, source: str = "auto") -> str:
        """
        source: "auto" | "wiki" | "ddg"
        """
        original_query = (query or "").strip()
        cleaned_query = self._simplify_query(original_query)
        source = (source or "auto").lower()

        # Optional tiny LLM rewrite to improve searchability for very noisy text
        if ENABLE_LLM_REWRITE:
            try:
                cleaned_query = await self._rewrite_query_with_llm(original_query, cleaned_query)
            except Exception as e:
                logger.warning(f"[SearchAgent] LLM rewrite failed, continuing without it: {e}")

        logger.info(f"[SearchAgent] search: source='{source}', original='{original_query}', cleaned='{cleaned_query}'")

        if source == "wiki":
            wiki_text, wiki_link = await self._search_wikipedia_with_resolution(original_query, cleaned_query)
            if wiki_text:
                summary = await self._summarize_with_llm("", wiki_text, original_query)
                return self._render(summary, wiki_link=wiki_link)
            ddg_text, ddg_link = await self._search_duckduckgo(cleaned_query)
            if ddg_text:
                summary = await self._summarize_with_llm(ddg_text, "", original_query)
                return self._render(summary, ddg_link=ddg_link)
            return self._no_info(original_query)

        if source == "ddg":
            ddg_text, ddg_link = await self._search_duckduckgo(cleaned_query)
            if ddg_text:
                summary = await self._summarize_with_llm(ddg_text, "", original_query)
                return self._render(summary, ddg_link=ddg_link)
            wiki_text, wiki_link = await self._search_wikipedia_with_resolution(original_query, cleaned_query)
            if wiki_text:
                summary = await self._summarize_with_llm("", wiki_text, original_query)
                return self._render(summary, wiki_link=wiki_link)
            return self._no_info(original_query)

        # AUTO: run both in parallel and combine
        ddg_task = asyncio.create_task(self._search_duckduckgo(cleaned_query))
        wiki_task = asyncio.create_task(self._search_wikipedia_with_resolution(original_query, cleaned_query))
        ddg_text, ddg_link = await ddg_task
        wiki_text, wiki_link = await wiki_task

        if not ddg_text and not wiki_text:
            return self._no_info(original_query)

        summary = await self._summarize_with_llm(ddg_text, wiki_text, original_query)
        return self._render(summary, ddg_link=ddg_link, wiki_link=wiki_link)

    # -------------------------------------------------------------------------
    # Engines
    # -------------------------------------------------------------------------
    async def _search_duckduckgo(self, query: str) -> Tuple[str, Optional[str]]:
        """
        Real web search using duckduckgo_search library.
        Returns top 3 results combined as (content, link).
        Falls back to the old Instant Answer API if library unavailable.
        """
        if _DDGS_AVAILABLE:
            return await self._search_ddgs(query)
        return await self._search_ddg_instant(query)

    async def _search_ddgs(self, query: str) -> Tuple[str, Optional[str]]:
        """
        Use duckduckgo_search DDGS for real web results (top 3 snippets).
        Runs in a thread pool since DDGS is synchronous.
        """
        def _run_search():
            try:
                with DDGS() as ddgs:
                    results = list(ddgs.text(query, max_results=3))
                return results
            except Exception as e:
                logger.warning(f"[SearchAgent] DDGS search error: {e}")
                return []

        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(None, _run_search)

        if not results:
            logger.info("[SearchAgent] DDGS: no results, falling back to instant API")
            return await self._search_ddg_instant(query)

        # Combine top snippets
        parts = []
        first_link = None
        for r in results:
            title = r.get("title", "")
            body  = r.get("body", "")
            href  = r.get("href", "")
            if body:
                parts.append(f"[{title}] {body}")
            if not first_link and href:
                first_link = href

        content = "\n\n".join(parts)
        logger.info(f"[SearchAgent] DDGS web results: {len(results)} hits, {len(content)} chars")

        # ── Relevance guard ──────────────────────────────────────────────────
        # If none of the query's meaningful keywords appear in the combined
        # snippets, the results are off-topic (rate-limit noise, geo-redirect,
        # etc.) — discard them so Wikipedia can be the sole source.
        _STOP = {"the", "a", "an", "of", "in", "on", "at", "is", "was",
                 "are", "were", "and", "or", "to", "for", "from", "by"}
        meaningful = [w for w in query.lower().split() if w not in _STOP and len(w) > 2]
        if meaningful:
            combined_lower = content.lower()
            hits = sum(1 for w in meaningful if w in combined_lower)
            if hits == 0:
                logger.warning(
                    f"[SearchAgent] DDG results irrelevant "
                    f"(0/{len(meaningful)} keywords matched) — discarding"
                )
                return ("", None)
        # ─────────────────────────────────────────────────────────────────────

        return (content, first_link)

    async def _search_ddg_instant(self, query: str) -> Tuple[str, Optional[str]]:
        """
        Legacy DuckDuckGo Instant Answer API (limited — only returns pre-indexed facts).
        Used as fallback when duckduckgo_search library is unavailable.
        """
        url = "https://api.duckduckgo.com/"
        params = {
            "q": query,
            "format": "json",
            "no_redirect": 1,
            "no_html": 1,
            "skip_disambig": 1,
        }
        headers = {"User-Agent": "Mozilla/5.0"}

        try:
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout, trust_env=True, headers=headers) as sess:
                async with sess.get(url, params=params) as resp:
                    if resp.status != 200:
                        logger.warning(f"[SearchAgent] DDG API status {resp.status}")
                        return ("", None)

                    data = await resp.json(content_type=None)

                    content = (
                        (data.get("AbstractText") or "").strip()
                        or (data.get("Definition") or "").strip()
                        or (data.get("Answer") or "").strip()
                    )

                    link = data.get("AbstractURL") or None

                    # Check related topics for fallback content/link
                    if (not content or not link) and data.get("RelatedTopics"):
                        for topic in data["RelatedTopics"]:
                            if isinstance(topic, dict):
                                if not content and topic.get("Text"):
                                    content = (content + " " + topic["Text"]).strip()
                                if not link and topic.get("FirstURL"):
                                    link = topic["FirstURL"]
                            if content and link:
                                break

                    # As a last resort, generic search link
                    if not link:
                        link = f"https://duckduckgo.com/?q={urllib.parse.quote(query)}"

                    logger.info(f"[SearchAgent] DDG chars={len(content)}")
                    return (content.strip(), link)
        except asyncio.TimeoutError:
            logger.error("[SearchAgent] DDG search timed out")
            return ("", None)
        except Exception as e:
            logger.error(f"[SearchAgent] DDG search error: {e}")
            return ("", None)

    async def _search_wikipedia_with_resolution(
        self, original_query: str, cleaned_query: str
    ) -> Tuple[str, Optional[str]]:
        """
        Resolve best page title via 'list=search', then fetch extract + canonical URL.
        Returns (content, page_url).
        """
        headers = {"User-Agent": "SearchAgent/1.0", "Accept": "application/json"}
        timeout = aiohttp.ClientTimeout(total=10)

        try:
            async with aiohttp.ClientSession(timeout=timeout, trust_env=True, headers=headers) as sess:
                # Step 1: resolve a good title (allow multiple; pick best with type-aware scoring)
                titles = await self._wiki_candidate_titles(sess, original_query, limit=8)
                if not titles:
                    titles = await self._wiki_candidate_titles(sess, cleaned_query, limit=8)

                # Heuristic aliases
                if not titles:
                    ql = cleaned_query.lower()
                    if any(tok in ql for tok in ["world war 1", "world war i", "ww1"]):
                        titles = ["World War I"]
                    elif any(tok in ql for tok in ["world war 2", "world war ii", "ww2"]):
                        titles = ["World War II"]

                if not titles:
                    logger.info("[SearchAgent] Wikipedia: no title candidates")
                    return ("", None)

                best = self._score_and_pick_title(cleaned_query, titles)

                # Step 2: extracts + canonical URL (string params!)
                extract_url = "https://en.wikipedia.org/w/api.php"
                extract_params = {
                    "action": "query",
                    "format": "json",
                    "titles": best,
                    "prop": "extracts|info",
                    "inprop": "url",
                    "explaintext": "1",
                    "exsectionformat": "plain",
                    "exchars": "2000",
                    "exlimit": "1",
                }
                async with sess.get(extract_url, params=extract_params) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        pages = data.get("query", {}).get("pages", {})
                        for page_id, page_data in pages.items():
                            if page_id != "-1" and "extract" in page_data:
                                content = (page_data.get("extract") or "").strip()
                                link = page_data.get("fullurl")
                                if content and len(content) > 80 and "may refer to:" not in content.lower():
                                    logger.info(f"[SearchAgent] Wikipedia extract chars={len(content)} title='{best}'")
                                    return (content, link)

                # Step 3: REST summary (short but reliable)
                encoded_title = urllib.parse.quote(best.replace(" ", "_"))
                summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded_title}"
                async with sess.get(summary_url) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        content = (data.get("extract") or "").strip()
                        link = data.get("content_urls", {}).get("desktop", {}).get("page")
                        if content and "may refer to:" not in content.lower():
                            return (content, link)

                return ("", None)

        except asyncio.TimeoutError:
            logger.error("[SearchAgent] Wikipedia timed out")
            return ("", None)
        except Exception as e:
            logger.error(f"[SearchAgent] Wikipedia search failed: {e}")
            return ("", None)

    async def _wiki_candidate_titles(self, sess: aiohttp.ClientSession, query: str, limit: int = 5) -> List[str]:
        """
        Use MediaWiki 'list=search' to get up to `limit` candidate titles.
        """
        q = self._strip_fillers(query or "")
        if not q:
            return []
        try:
            search_url = "https://en.wikipedia.org/w/api.php"
            params = {
                "action": "query",
                "format": "json",
                "list": "search",
                "srsearch": q,
                "srlimit": str(limit),
            }
            async with sess.get(search_url, params=params) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()
                hits = data.get("query", {}).get("search", [])
                return [h.get("title") for h in hits if h.get("title")]
        except Exception as e:
            logger.warning(f"[SearchAgent] _wiki_candidate_titles error: {e}")
            return []

    # -------------------------------------------------------------------------
    # LLM helpers
    # -------------------------------------------------------------------------
    def _llm_available(self) -> bool:
        return self._llm_client is not None

    async def _llm_one_shot(self, prompt: str) -> str:
        """
        Call the model client directly for a single LLM response.
        Bypasses AutoGen team/stream machinery entirely — much simpler and
        avoids any risk of the user-task message leaking into the output.
        """
        if not self._llm_client:
            return ""
        try:
            from autogen_core.models import UserMessage
            result = await self._llm_client.create(
                messages=[UserMessage(content=prompt, source="user")]
            )
            content = result.content if hasattr(result, "content") else ""
            if isinstance(content, list):          # FunctionCall list edge-case
                content = " ".join(str(c) for c in content)
            return (content or "").strip()
        except Exception as e:
            logger.error(f"[SearchAgent] _llm_one_shot error: {e}")
            return ""

    async def _rewrite_query_with_llm(self, original: str, cleaned: str) -> str:
        if not self._llm_available():
            return cleaned

        prompt = f"""
Rewrite the user's question into a short literal web search query (max 6 words), preserving meaning and key entities.
- Do not add quotes or special syntax.
- Return only the query text, nothing else.

User question: {original!r}
Initial cleaned: {cleaned!r}
""".strip()
        try:
            text = await self._llm_one_shot(prompt)
            text = (text or "").strip().strip('"').strip("'")
            return text[:120] or cleaned
        except Exception as e:
            logger.warning(f"[SearchAgent] rewrite failed: {e}")
            return cleaned

    async def _summarize_with_llm(self, ddg_text: str, wiki_text: str, original_query: str) -> str:
        if not self._llm_available():
            # graceful fallback to raw content if LLM not available
            parts = []
            if ddg_text:
                parts.append(f"DuckDuckGo: {ddg_text}")
            if wiki_text:
                parts.append(f"Wikipedia: {wiki_text}")
            return "\n\n".join(parts) if parts else "❌ Both search and summarization failed."

        sources_text = ""
        if ddg_text:
            sources_text += f"Source 1 (DuckDuckGo):\n{ddg_text}\n\n"
        if wiki_text:
            sources_text += f"Source 2 (Wikipedia):\n{wiki_text}\n\n"

        if not sources_text.strip():
            return "No information available from search sources."

        prompt = f"""You are answering the user's question: "{original_query}"

Based only on the factual information below, write a concise, accurate answer.
- Prefer details that both sources agree on.
- Acknowledge uncertainty if facts conflict.
- Avoid speculation.

{sources_text}

Answer in 3–6 sentences:"""
        try:
            return await self._llm_one_shot(prompt)
        except Exception as e:
            logger.error(f"[SearchAgent] LLM summarization failed: {e}")
            # Fallback: return raw content
            parts = []
            if ddg_text:
                parts.append(f"DuckDuckGo: {ddg_text}")
            if wiki_text:
                parts.append(f"Wikipedia: {wiki_text}")
            return "\n\n".join(parts) if parts else "❌ Both search and summarization failed."

    # -------------------------------------------------------------------------
    # Utils
    # -------------------------------------------------------------------------
    def _simplify_query(self, query: str) -> str:
        cleaned = (query or "").strip().lower()

        # remove leading question phrases
        question_phrases = [
            "what is", "who is", "who was", "when did", "when was",
            "tell me about", "information about", "can you", "could you",
            "would you", "please", "explain", "in brief", "briefly",
            "give me", "give", "provide",
        ]
        for qw in question_phrases:
            if cleaned.startswith(qw):
                cleaned = cleaned[len(qw):].strip()

        if cleaned.endswith("?"):
            cleaned = cleaned[:-1].strip()

        cleaned = self._strip_fillers(cleaned)

        # normalize common aliases for WWI/WWII
        cleaned = cleaned.replace("world war 1", "world war i").replace("ww1", "world war i")
        cleaned = cleaned.replace("world war 2", "world war ii").replace("ww2", "world war ii")
        
        # normalize common misspellings/variants for historical terms
        cleaned = cleaned.replace("magadh", "magadha")  # ancient Indian kingdom
        
        return cleaned

    def _strip_fillers(self, text: str) -> str:
        t = text.strip()
        # remove multi-word phrases first
        for ph in ["in brief", "tell me", "give me"]:
            t = t.replace(ph, " ")
        words = [w for w in t.split() if w not in _STOP_TERMS]
        return " ".join(words).strip()

    def _score_and_pick_title(self, query: str, candidates: List[str]) -> str:
        """
        Prefer exact/short/canonical pages, de-prioritize institutions/film/TV/list pages.
        
        Scoring heuristics:
        - Word overlap: +1 per matching word
        - Exact match: +5
        - Starts with query: +3
        - Short, simple titles: +2 (canonical articles tend to be short)
        - No parentheses (canonical page): +1
        - Long institution names (hospital, college, university): -4
        - Media pages (film, TV, etc.): -3
        """
        q = query.lower()
        qwords = set(q.split())
        best = candidates[0]
        best_score = -999

        for title in candidates:
            tl = title.lower()
            twords = set(tl.split())
            score = 0

            # Word overlap
            overlap = len(qwords & twords)
            score += overlap

            # Exact match is ideal
            if tl == q:
                score += 5
            # Starts with query (e.g., "Magadha" for query "magadha")
            elif tl.startswith(q) or q.startswith(tl):
                score += 3

            # Penalize long titles (institutions, full names with qualifiers)
            # Canonical wiki pages for concepts are usually short (1-3 words)
            word_count = len(twords)
            if word_count <= 2:
                score += 2
            elif word_count >= 5:
                score -= 2
            
            # Heavily penalize institutional/organization pages
            institution_markers = [
                "hospital", "college", "university", "school", "institute",
                "academy", "foundation", "museum", "railway", "station",
                "district", "constituency", "assembly", "airport"
            ]
            if any(m in tl for m in institution_markers):
                score -= 4

            # Penalize media/entertainment pages
            media_markers = ["film", "tv", "television", "series", "episode", 
                           "novel", "song", "album", "video game", "movie"]
            if any(m in tl for m in media_markers):
                score -= 3

            # Penalize list/disambiguation pages
            if tl.startswith("list of") or "disambiguation" in tl:
                score -= 5

            # Bonus for canonical (no parentheses) pages
            if "(" not in tl:
                score += 1

            if score > best_score:
                best_score = score
                best = title

        return best

    def _render(self, summary: str, ddg_link: Optional[str] = None, wiki_link: Optional[str] = None) -> str:
        sources = []
        if ddg_link:
            sources.append(f"DuckDuckGo: {ddg_link}")
        if wiki_link:
            sources.append(f"Wikipedia: {wiki_link}")
        src_block = "\n\n🔗 Sources:\n" + "\n".join(f"- {s}" for s in sources) if sources else ""
        return f"📘 Summary:\n{summary}{src_block}"

    def _no_info(self, q: str) -> str:
        return f"ℹ️ No useful information found for \"{q}\" from either DuckDuckGo or Wikipedia."
