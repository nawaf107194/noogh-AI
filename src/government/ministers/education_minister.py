#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Education Minister - System Strategist & Optimizer
===================================================

Analyzes data from other ministers and generates optimization strategies.
Acts as the "brain" of the system - learning and improving strategies.
"""

from typing import Optional, Dict, Any
import logging

from .base_minister import BaseMinister

logger = logging.getLogger(__name__)


class EducationMinister(BaseMinister):
    """
    Minister of Education - System Strategist and Optimizer.
    
    New Role: Analyzes data from Finance/Security ministers and generates
    algorithmic strategies, rules, and system improvements.
    """
    
    def __init__(self, brain: Optional[Any] = None):
        """Initialize Education Minister."""
        super().__init__(
            name="Education Minister (Strategist)",
            description="System architect and strategy optimizer. Generates algorithmic improvements.",
            brain=brain
        )
        
        self.system_prompt = """You are the System Architect and Strategic Optimizer.
Your role is to analyze data from various system components and formulate:
- Algorithmic trading strategies
- Security mitigation strategies
- System optimization rules
- Risk management frameworks

Be specific, actionable, and provide concrete implementation steps.
Think like a senior systems architect designing robust, production-ready solutions."""
    
    async def generate_strategy(
        self,
        data_type: str,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate an optimization strategy based on input data.
        
        Args:
            data_type: Type of data (market, security, performance)
            data: The data to analyze
        
        Returns:
            Strategy recommendations
        """
        try:
            # Prepare prompt based on data type
            if data_type == "market":
                prompt = f"""Market Analysis Data:
{self._format_dict(data)}

Based on this market data, formulate:
1. A specific algorithmic trading strategy
2. Entry/exit rules
3. Position sizing recommendations
4. Stop-loss and take-profit levels
5. Risk management parameters

Provide a concrete, implementable strategy."""
            
            elif data_type == "security":
                prompt = f"""Security Threat Data:
{self._format_dict(data)}

Based on this security analysis, formulate:
1. Specific mitigation steps
2. System hardening recommendations
3. Monitoring rules to implement
4. Prevention strategies
5. Incident response procedures

Provide actionable security improvements."""
            
            else:
                prompt = f"""System Data:
{self._format_dict(data)}

Analyze this data and provide:
1. Key insights
2. Optimization opportunities
3. Recommended improvements
4. Implementation steps

Be specific and actionable."""
            
            # Get AI strategy
            strategy = await self._think_with_prompt(
                system_prompt=self.system_prompt,
                user_message=prompt,
                max_tokens=800
            )
            
            return {
                "success": True,
                "strategy": strategy,
                "data_type": data_type
            }
        
        except Exception as e:
            logger.error(f"Strategy generation error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _format_dict(self, data: Dict[str, Any]) -> str:
        """Format dictionary for prompt."""
        lines = []
        for key, value in data.items():
            if isinstance(value, dict):
                lines.append(f"{key}:")
                for k, v in value.items():
                    lines.append(f"  {k}: {v}")
            else:
                lines.append(f"{key}: {value}")
        return "\n".join(lines)
    
    async def execute_task(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute strategy generation task.
        
        Args:
            task: Task description
            context: Optional context with data to analyze
        
        Returns:
            Strategy recommendations
        """
        self.tasks_processed += 1
        
        try:
            # Check if context has data from other ministers
            if context and "minister_data" in context:
                data = context["minister_data"]
                data_type = context.get("data_type", "general")
                
                result = await self.generate_strategy(data_type, data)
                
                if result.get("success"):
                    self.tasks_successful += 1
                    
                    return {
                        "success": True,
                        "response": result['strategy'],
                        "minister": self.name,
                        "domain": "strategy",
                        "metadata": {
                            "data_type": data_type
                        }
                    }
            
            # Fallback to general educational response
            response = await self._think_with_prompt(
                system_prompt=self.system_prompt,
                user_message=task,
                max_tokens=600
            )
            
            self.tasks_successful += 1
            
            return {
                "success": True,
                "response": response,
                "minister": self.name,
                "domain": "education",
                "metadata": {}
            }
        
        except Exception as e:
            logger.error(f"Education Minister error: {e}")
            return {
                "success": False,
                "response": f"Strategy generation failed: {str(e)}",
                "minister": self.name,
                "error": str(e)
            }


# ============================================================================
# Exports
# ============================================================================

__all__ = ["EducationMinister"]
