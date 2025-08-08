# agents/search_agent.py
import aiohttp
import asyncio
import urllib.parse
import logging
from autogen_agentchat.agents import AssistantAgent

# Set up logging to debug issues
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SearchAgent(AssistantAgent):
    """
    Agent that performs factual web lookups using DuckDuckGo and Wikipedia,
    then summarizes the result using an LLM (model_client).
    """

    def __init__(self, name="search", model_client=None):
        super().__init__(
            name=name,
            model_client=model_client,
            system_message=(
                "You are the SearchAgent. Given a query, you fetch factual information from the DuckDuckGo "
                "Instant Answer API and Wikipedia, then generate a natural and concise summary using the LLM. "
                "Avoid simulating sources. Prioritize real-world facts."
            )
        )

    async def search(self, query: str) -> str:
        cleaned_query = self._simplify_query(query)
        logger.info(f"Searching for: original='{query}', cleaned='{cleaned_query}'")
        
        ddg_result, wiki_result = await self._run_parallel_searches(cleaned_query, query)
        
        # Handle exceptions from parallel searches
        if isinstance(ddg_result, Exception):
            logger.error(f"DDG search failed with exception: {ddg_result}")
            ddg_result = ""
        if isinstance(wiki_result, Exception):
            logger.error(f"Wikipedia search failed with exception: {wiki_result}")
            wiki_result = ""
        
        logger.info(f"DDG result length: {len(ddg_result) if ddg_result else 0}")
        logger.info(f"Wiki result length: {len(wiki_result) if wiki_result else 0}")
        
        # Debug: Print first 200 chars of each result
        if ddg_result:
            logger.info(f"DDG preview: {ddg_result[:200]}...")
        if wiki_result:
            logger.info(f"Wiki preview: {wiki_result[:200]}...")

        if not ddg_result and not wiki_result:
            return f"ℹ️ No useful information found for \"{query}\" from either DuckDuckGo or Wikipedia."

        summary = await self._summarize_with_llm(ddg_result, wiki_result, query)
        return f"📘 Summary:\n{summary}"

    async def _run_parallel_searches(self, ddg_query: str, wiki_query: str):
        return await asyncio.gather(
            self._search_duckduckgo(ddg_query),
            self._search_wikipedia(wiki_query),
            return_exceptions=True
        )

    async def _search_duckduckgo(self, query: str) -> str:
        url = "https://api.duckduckgo.com/"
        params = {
            "q": query,
            "format": "json",
            "no_redirect": 1,
            "no_html": 1,
            "skip_disambig": 1
        }
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        try:
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(
                timeout=timeout,
                trust_env=True,
                headers=headers
            ) as sess:
                async with sess.get(url, params=params) as resp:
                    if resp.status != 200:
                        logger.warning(f"DDG API returned status {resp.status}")
                        return ""
                    
                    data = await resp.json(content_type=None)
                    
                    # Try multiple fields for content
                    content = (
                        data.get("AbstractText", "").strip() or
                        data.get("Definition", "").strip() or
                        data.get("Answer", "").strip()
                    )
                    
                    # Also check related topics if main content is empty
                    if not content and data.get("RelatedTopics"):
                        for topic in data.get("RelatedTopics", [])[:2]:  # Take first 2
                            if isinstance(topic, dict) and topic.get("Text"):
                                content += topic["Text"] + " "
                    
                    logger.info(f"DDG content found: {len(content)} chars")
                    return content.strip()
                    
        except asyncio.TimeoutError:
            logger.error("DDG search timed out")
            return ""
        except Exception as e:
            logger.error(f"DDG search error: {e}")
            return ""

    async def _search_wikipedia(self, query: str) -> str:
        # Clean and encode the query properly
        clean_query = query.replace(" biography", "").strip()
        
        headers = {
            'User-Agent': 'SearchAgent/1.0 (https://example.com/contact)',
            'Accept': 'application/json'
        }
        
        timeout = aiohttp.ClientTimeout(total=10)
        
        try:
            async with aiohttp.ClientSession(timeout=timeout, trust_env=True, headers=headers) as sess:
                # Method 1: Try Wikipedia extract API for more comprehensive info including personal details
                try:
                    extract_url = "https://en.wikipedia.org/w/api.php"
                    extract_params = {
                        'action': 'query',
                        'format': 'json',
                        'titles': clean_query,
                        'prop': 'extracts',
                        'exintro': False,  # Get full article, not just intro
                        'explaintext': True,
                        'exsectionformat': 'plain',
                        'exchars': 1500,  # Get more content to include personal details
                        'exlimit': 1
                    }
                    
                    async with sess.get(extract_url, params=extract_params) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            pages = data.get('query', {}).get('pages', {})
                            for page_id, page_data in pages.items():
                                if page_id != '-1' and 'extract' in page_data:
                                    content = page_data['extract'].strip()
                                    if content and len(content) > 100 and "may refer to:" not in content.lower():
                                        logger.info(f"Wikipedia extract API success: {len(content)} chars")
                                        return content
                except Exception as e:
                    logger.warning(f"Wikipedia extract API failed: {e}")
                
                # Method 2: Try to get the full page content with specific sections
                try:
                    sections_url = "https://en.wikipedia.org/w/api.php"
                    sections_params = {
                        'action': 'query',
                        'format': 'json',
                        'titles': clean_query,
                        'prop': 'extracts',
                        'explaintext': True,
                        'exsectionformat': 'plain',
                        'exchars': 2000  # Even more content
                    }
                    
                    async with sess.get(sections_url, params=sections_params) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            pages = data.get('query', {}).get('pages', {})
                            for page_id, page_data in pages.items():
                                if page_id != '-1' and 'extract' in page_data:
                                    content = page_data['extract'].strip()
                                    if content and len(content) > 100:
                                        logger.info(f"Wikipedia sections API success: {len(content)} chars")
                                        return content
                except Exception as e:
                    logger.warning(f"Wikipedia sections API failed: {e}")
                
                # Method 3: Fallback to REST API summary
                try:
                    encoded_query = urllib.parse.quote(clean_query.replace(' ', '_'))
                    summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded_query}"
                    
                    async with sess.get(summary_url) as resp:
                        logger.info(f"Wikipedia REST API status: {resp.status} for query: {clean_query}")
                        
                        if resp.status == 200:
                            data = await resp.json()
                            content = data.get("extract", "").strip()
                            
                            if content and "may refer to:" not in content.lower():
                                logger.info(f"Wikipedia REST API success: {len(content)} chars")
                                return content
                        elif resp.status == 404 and clean_query != query:
                            # Try with original query
                            encoded_original = urllib.parse.quote(query.replace(' ', '_'))
                            url_original = f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded_original}"
                            async with sess.get(url_original) as resp2:
                                if resp2.status == 200:
                                    data = await resp2.json()
                                    content = data.get("extract", "").strip()
                                    if content:
                                        logger.info(f"Wikipedia found with original query: {len(content)} chars")
                                        return content
                except Exception as e:
                    logger.warning(f"Wikipedia REST API failed: {e}")
                
                return ""
                    
        except Exception as e:
            logger.error(f"Wikipedia search completely failed: {e}")
            return ""

    async def _summarize_with_llm(self, ddg_text: str, wiki_text: str, original_query: str) -> str:
        """
        Use LLM (model_client) to summarize and deduplicate DDG and Wikipedia responses.
        """
        # Prepare sources text
        sources_text = ""
        if ddg_text:
            sources_text += f"Source 1 (DuckDuckGo):\n{ddg_text}\n\n"
        if wiki_text:
            sources_text += f"Source 2 (Wikipedia):\n{wiki_text}\n\n"
        
        if not sources_text.strip():
            return "No information available from search sources."
        
        # Analyze query to identify what specific information is being asked
        query_lower = original_query.lower()
        specific_requests = []
        if any(word in query_lower for word in ['born', 'birth', 'birthday']):
            specific_requests.append("birth date")
        if any(word in query_lower for word in ['married', 'marriage', 'wife', 'spouse']):
            specific_requests.append("marriage information")
        if any(word in query_lower for word in ['who is', 'about', 'biography']):
            specific_requests.append("general biography")
        
        prompt = f"""You are answering the user's question: "{original_query}"

Based on the factual information provided below, answer the user's specific questions comprehensively.

The user is asking for: {', '.join(specific_requests) if specific_requests else 'general information'}

IMPORTANT INSTRUCTIONS:
- If the user asks about birth/birthday, look for and include birth date information
- If the user asks about marriage/married, look for and include marriage/spouse information  
- If the user asks "who is", provide a comprehensive biography
- Include specific dates, names, and details when available
- If some requested information is not available in the sources, mention that clearly

{sources_text}

Now provide a complete answer to "{original_query}" that addresses all parts of the user's question:"""
        
        try:
            response = await self.model_client.aask(prompt)
            return response.strip()
        except Exception as e:
            logger.error(f"LLM summarization failed: {e}")
            # Fallback: return the raw content if LLM fails
            fallback_content = []
            if ddg_text:
                fallback_content.append(f"DuckDuckGo: {ddg_text}")
            if wiki_text:
                fallback_content.append(f"Wikipedia: {wiki_text}")
            return "\n\n".join(fallback_content) if fallback_content else "❌ Both search and summarization failed."

    def _simplify_query(self, query: str) -> str:
        """
        Simplify/clean the user query for use in search APIs.
        """
        # Remove common stop words that might interfere with search
        cleaned = query.strip().lower()
        
        # Remove question words
        question_words = ["what is", "who is", "tell me about", "information about"]
        for qw in question_words:
            if cleaned.startswith(qw):
                cleaned = cleaned[len(qw):].strip()
        
        # Remove trailing question mark
        if cleaned.endswith("?"):
            cleaned = cleaned[:-1].strip()
        
        return cleaned