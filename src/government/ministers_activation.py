#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Noogh Government System - Ministers Activation
نظام الحكومة الداخلية لنوغ - تفعيل الوزراء
"""
import logging
from typing import Dict, Any, Optional

from .minister_types_universal import MinisterType
from .base_minister import generate_task_id

# Import all minister implementations
from .education_minister import EducationMinister
from .security_minister import SecurityMinister
from .development_minister import DevelopmentMinister
from .research_minister import ResearchMinister
from .knowledge_minister import KnowledgeMinister
from .privacy_minister import PrivacyMinister
from .creativity_minister import CreativityMinister
from .analysis_minister import AnalysisMinister
from .strategy_minister import StrategyMinister
from .reasoning_minister import ReasoningMinister
from .communication_minister import CommunicationMinister
from .resources_minister import ResourcesMinister
from .finance_minister import FinanceMinister
# Note: Training minister is not in the file list, so it's commented out.
# from .training_minister import TrainingMinister 

logger = logging.getLogger(__name__)

class MinistersActivationSystem:
    """
    A system to activate and manage all ministers in the government.
    """
    def __init__(self, brain_hub: Any, verbose: bool = True):
        """
        Initializes the ministers activation system.
        Args:
            brain_hub: A reference to the UnifiedBrainHub for ministers to use.
            verbose: Enable detailed logging.
        """
        self.brain_hub = brain_hub
        self.verbose = verbose
        self.active_ministers: Dict[MinisterType, Any] = {}
        self.minister_map = {
            MinisterType.EDUCATION: EducationMinister,
            MinisterType.SECURITY: SecurityMinister,
            MinisterType.DEVELOPMENT: DevelopmentMinister,
            MinisterType.RESEARCH: ResearchMinister,
            MinisterType.KNOWLEDGE: KnowledgeMinister,
            MinisterType.PRIVACY: PrivacyMinister,
            MinisterType.CREATIVITY: CreativityMinister,
            MinisterType.ANALYSIS: AnalysisMinister,
            MinisterType.STRATEGY: StrategyMinister,
            MinisterType.REASONING: ReasoningMinister,
            MinisterType.COMMUNICATION: CommunicationMinister,
            MinisterType.RESOURCES: ResourcesMinister,
            MinisterType.FINANCE: FinanceMinister,
            # MinisterType.TRAINING: TrainingMinister,
        }

    def activate_all(self):
        """
        Activates and instantiates all available ministers.
        """
        if self.active_ministers:
            logger.info("Ministers are already active.")
            return

        logger.info("Activating all government ministers...")
        for minister_type, minister_class in self.minister_map.items():
            try:
                # Pass the brain_hub (or a proxy) to the minister if its constructor accepts it
                # This is a common pattern for dependency injection in agent systems
                minister_instance = minister_class(brain_hub=self.brain_hub, verbose=self.verbose)
                self.active_ministers[minister_type] = minister_instance
                if self.verbose:
                    logger.info(f"✅ Minister {minister_type.value.capitalize()} activated.")
            except Exception as e:
                logger.error(f"❌ Failed to activate minister {minister_type.value}: {e}")
        
        logger.info(f"Total active ministers: {len(self.active_ministers)}")

    async def delegate_task(self, minister_type: MinisterType, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Delegates a task to the specified minister.
        Args:
            minister_type: The type of minister to delegate the task to.
            task: The task data dictionary.
        Returns:
            A dictionary with the result from the minister.
        """
        minister = self.active_ministers.get(minister_type)
        if not minister:
            logger.error(f"No active minister found for type: {minister_type.value}")
            return {"error": f"Minister {minister_type.value} not found or not active."}

        task_id = generate_task_id()
        task_type = task.get("type", "general_request")
        
        try:
            report = await minister.execute_task(task_id, task_type, task)
            return report.to_dict()
        except Exception as e:
            logger.error(f"Error delegating task to {minister_type.value}: {e}")
            return {"error": str(e), "minister": minister_type.value}

    def get_all_stats(self) -> Dict[str, Any]:
        """
        Gathers status reports from all active ministers.
        Returns:
            A dictionary containing the stats of all ministers.
        """
        stats = {}
        for minister_type, minister in self.active_ministers.items():
            stats[minister_type.value] = minister.get_status_report()
        return stats