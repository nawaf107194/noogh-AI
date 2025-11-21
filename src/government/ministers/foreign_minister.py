#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Foreign Affairs Minister - News Scout & Trading Trigger
========================================================

Monitors crypto news and triggers Finance Minister analysis.
"""

from typing import Optional, Dict, Any, List
import logging
import re

from .base_minister import BaseMinister

logger = logging.getLogger(__name__)


class ForeignMinister(BaseMinister):
    """
    Foreign Minister - News Intelligence + Trading Trigger.
    
    NEW Powers:
    - Crypto news monitoring
    - High-impact event detection
    - Direct Finance Minister triggers
    """
    
    def __init__(self, brain: Optional[Any] = None):
        """Initialize Foreign Minister."""
        super().__init__(
            name="Foreign Minister (News Scout)",
            description="Crypto news monitor. Triggers trades on high-impact events.",
            brain=brain
        )
        
        self.system_prompt = """You are a crypto news analyst.
Detect high-impact events that could move prices:
- Regulatory news
- Major partnerships
- Exchange listings
- Protocol upgrades

Extract the affected coin symbol and impact level (HIGH/MEDIUM/LOW)."""
    
    def search_crypto_news(self, query: str = "cryptocurrency news") -> List[Dict[str, str]]:
        """Search for crypto news."""
        try:
            from googlesearch import search
            
            results = []
            for url in search(query, num_results=5, lang="en"):
                results.append({"url": url, "query": query})
            
            return results
        except Exception as e:
            logger.error(f"News search error: {e}")
            return []
    
    async def monitor_news_for_triggers(self) -> Dict[str, Any]:
        """
        Monitor news and detect trading triggers.
        
        Returns:
            Detected triggers with coin symbols
        """
        logger.info("📰 Monitoring crypto news...")
        
        try:
            # Search for recent crypto news
            news_results = self.search_crypto_news("latest cryptocurrency news today")
            
            if not news_results:
                return {
                    "success": True,
                    "triggers": [],
                    "message": "No news sources available"
                }
            
            triggers = []
            
            # Analyze each news item
            for news_item in news_results[:3]:  # Check top 3
                url = news_item['url']
                
                # Scrape page content
                page_data = self.read_page(url, max_chars=2000)
                
                if not page_data.get("success"):
                    continue
                
                text = page_data.get("text", "")
                title = page_data.get("title", "")
                
                # Use AI to detect high-impact events
                analysis = await self._analyze_news_impact(title, text)
                
                if analysis.get("impact") == "HIGH" and analysis.get("symbol"):
                    triggers.append({
                        "symbol": analysis["symbol"],
                        "impact": "HIGH",
                        "reason": analysis.get("reason", "News event"),
                        "source": url
                    })
                    
                    logger.warning(f"🚨 HIGH IMPACT NEWS: {analysis['symbol']} - {analysis.get('reason')}")
            
            return {
                "success": True,
                "triggers": triggers,
                "news_checked": len(news_results)
            }
        
        except Exception as e:
            logger.error(f"News monitoring error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _analyze_news_impact(self, title: str, text: str) -> Dict[str, Any]:
        """
        Analyze news for trading impact.
        
        Args:
            title: News title
            text: News content
        
        Returns:
            Impact analysis
        """
        prompt = f"""Analyze this crypto news:

Title: {title}
Content: {text[:500]}...

Tasks:
1. Impact Level: HIGH, MEDIUM, or LOW
2. Affected Coin Symbol (e.g., BTC, ETH, XRP) or NONE
3. Brief Reason

Format:
IMPACT: [level]
SYMBOL: [symbol]
REASON: [reason]"""
        
        response = await self._think_with_prompt(
            system_prompt=self.system_prompt,
            user_message=prompt,
            max_tokens=150
        )
        
        # Parse response
        impact = "LOW"
        symbol = None
        reason = "General news"
        
        if "IMPACT: HIGH" in response.upper():
            impact = "HIGH"
        elif "IMPACT: MEDIUM" in response.upper():
            impact = "MEDIUM"
        
        # Extract symbol
        symbol_match = re.search(r'SYMBOL:\s*([A-Z]+)', response)
        if symbol_match:
            raw_symbol = symbol_match.group(1)
            if raw_symbol not in ["NONE", "NA", "N/A"]:
                symbol = f"{raw_symbol}/USDT"
        
        # Extract reason
        reason_match = re.search(r'REASON:\s*(.+)', response, re.IGNORECASE)
        if reason_match:
            reason = reason_match.group(1).strip()
        
        return {
            "impact": impact,
            "symbol": symbol,
            "reason": reason
        }
    
    def search_web(self, query: str, num_results: int = 5) -> List[Dict[str, str]]:
        """Search web (original method)."""
        try:
            from googlesearch import search
            
            results = []
            for url in search(query, num_results=num_results, lang="en"):
                results.append({"url": url, "query": query})
            
            return results
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []
    
    def read_page(self, url: str, max_chars: int = 5000) -> Dict[str, Any]:
        """Scrape web page (original method)."""
        try:
            import requests
            from bs4 import BeautifulSoup
            
            headers = {'User-Agent': 'Mozilla/5.0 (compatible; NooghBot/1.0)'}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            for script in soup(["script", "style"]):
                script.decompose()
            
            text = soup.get_text(separator=' ', strip=True)
            
            if len(text) > max_chars:
                text = text[:max_chars] + "..."
            
            return {
                "success": True,
                "url": url,
                "text": text,
                "title": soup.title.string if soup.title else url
            }
        except Exception as e:
            logger.error(f"Scraping error: {e}")
            return {"success": False, "error": str(e)}
    
    async def execute_task(self, task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute intelligence task."""
        self.tasks_processed += 1
        
        try:
            task_lower = task.lower()
            
            # News monitoring
            if "news" in task_lower or "monitor" in task_lower:
                result = await self.monitor_news_for_triggers()
                
                if result.get("success"):
                    self.tasks_successful += 1
                    
                    triggers = result.get("triggers", [])
                    
                    if triggers:
                        summary = f"🚨 Found {len(triggers)} high-impact news triggers:\n\n"
                        for trigger in triggers:
                            summary += f"- {trigger['symbol']}: {trigger['reason']}\n"
                    else:
                        summary = f"Monitored {result.get('news_checked', 0)} news sources. No high-impact triggers."
                    
                    return {
                        "success": True,
                        "response": summary,
                        "minister": self.name,
                        "domain": "intelligence",
                        "metadata": result
                    }
            
            # Standard web search (original)
            else:
                search_results = self.search_web(task, num_results=3)
                
                if not search_results:
                    return {
                        "success": False,
                        "response": "Web search unavailable",
                        "minister": self.name
                    }
                
                page_content = self.read_page(search_results[0]['url'])
                
                if not page_content.get("success"):
                    intelligence_data = f"Search results only (no scraping)"
                else:
                    intelligence_data = f"{page_content['title']}\n\n{page_content['text'][:500]}..."
                
                analysis = await self._think_with_prompt(
                    system_prompt="You are a strategic intelligence analyst.",
                    user_message=f"Analyze this: {intelligence_data}",
                    max_tokens=400
                )
                
                self.tasks_successful += 1
                
                return {
                    "success": True,
                    "response": analysis,
                    "minister": self.name,
                    "domain": "intelligence",
                    "metadata": {"search_results": search_results}
                }
        
        except Exception as e:
            logger.error(f"Foreign Minister error: {e}")
            return {
                "success": False,
                "response": str(e),
                "minister": self.name,
                "error": str(e)
            }


__all__ = ["ForeignMinister"]
