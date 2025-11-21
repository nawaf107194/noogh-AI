import time
import re
from enum import Enum
from typing import NamedTuple, List, Dict, Any, Optional

class Intent(Enum):
    QUESTION_KB = "question.knowledge_base"
    QUESTION_WEB = "question.web_search"
    CHITCHAT = "chitchat"
    COMMAND_NOT_IMPLEMENTED = "command.not_implemented"
    UNKNOWN = "unknown"

class RouterResponse(NamedTuple):
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    intent: Intent
    handler_used: str
    processing_time: float
    metadata: Dict[str, Any] = {}

class IntentRouter:
    def __init__(self, knowledge_kernel, enable_web_search: bool = True):
        self.knowledge_kernel = knowledge_kernel
        self.enable_web_search = enable_web_search

    def _classify_intent(self, question: str) -> Intent:
        """Classifies the intent of the user's question."""
        question_lower = question.lower()

        # Keywords for different intents
        question_keywords = ["what", "who", "where", "when", "why", "how", "explain", "define"]
        chitchat_keywords = ["hello", "hi", "how are you", "thanks", "thank you"]
        command_keywords = ["buy", "sell", "trade", "analyze", "run"]

        if any(keyword in question_lower for keyword in command_keywords):
            return Intent.COMMAND_NOT_IMPLEMENTED
        
        if any(keyword in question_lower for keyword in question_keywords):
            # For now, we'll default to web search for questions, 
            # as we don't have a good way to know if the KB has the answer.
            return Intent.QUESTION_WEB if self.enable_web_search else Intent.QUESTION_KB

        if any(keyword in question_lower for keyword in chitchat_keywords):
            return Intent.CHITCHAT

        return Intent.UNKNOWN

    def _handle_knowledge_question(self, question: str) -> RouterResponse:
        """Handles questions that should be answered by the internal knowledge base."""
        start_time = time.time()
        search_results = self.knowledge_kernel.search_knowledge(question, top_k=3)
        
        if not search_results:
            return self._handle_web_question(question) if self.enable_web_search else RouterResponse(
                answer="I could not find an answer in my knowledge base.",
                sources=[],
                confidence=0.1,
                intent=Intent.QUESTION_KB,
                handler_used="_handle_knowledge_question",
                processing_time=time.time() - start_time
            )

        context = "\n".join([res.get('full_chunk', res.get('chunk', '')) for res in search_results])
        answer = f"From my knowledge base:\n\n{context}"
        sources = [{"path": res.get("path")} for res in search_results if res.get("path")]
        
        return RouterResponse(
            answer=answer,
            sources=sources,
            confidence=0.7,
            intent=Intent.QUESTION_KB,
            handler_used="_handle_knowledge_question",
            processing_time=time.time() - start_time
        )

    def _handle_web_question(self, question: str) -> RouterResponse:
        """Handles questions that require a web search."""
        start_time = time.time()
        if not self.knowledge_kernel.web_search:
            return RouterResponse(
                answer="Web search is not available.",
                sources=[],
                confidence=0.0,
                intent=Intent.QUESTION_WEB,
                handler_used="_handle_web_question",
                processing_time=time.time() - start_time
            )
            
        search_results = self.knowledge_kernel.web_search.search(question, max_results=3)

        if not search_results:
            return RouterResponse(
                answer="I couldn't find anything on the web for that query.",
                sources=[],
                confidence=0.1,
                intent=Intent.QUESTION_WEB,
                handler_used="_handle_web_question",
                processing_time=time.time() - start_time
            )

        snippets = [f"{i+1}. {res['snippet']}" for i, res in enumerate(search_results)]
        answer = "According to a web search:\n\n" + "\n".join(snippets)
        
        return RouterResponse(
            answer=answer,
            sources=search_results,
            confidence=0.6,
            intent=Intent.QUESTION_WEB,
            handler_used="_handle_web_question",
            processing_time=time.time() - start_time,
            metadata={"used_web_search": True}
        )

    def _handle_chitchat(self, question: str) -> RouterResponse:
        """Handles simple chitchat."""
        start_time = time.time()
        answer = "Hello! How can I help you today?"
        if "how are you" in question.lower():
            answer = "I am an AI, I don't have feelings, but I'm operating at full capacity. Thanks for asking!"
        elif "thank" in question.lower():
            answer = "You're welcome!"
            
        return RouterResponse(
            answer=answer,
            sources=[],
            confidence=0.9,
            intent=Intent.CHITCHAT,
            handler_used="_handle_chitchat",
            processing_time=time.time() - start_time
        )
        
    def _handle_not_implemented(self, question: str) -> RouterResponse:
        """Handles commands that are not yet implemented."""
        start_time = time.time()
        return RouterResponse(
            answer="This command is not yet implemented.",
            sources=[],
            confidence=0.8,
            intent=Intent.COMMAND_NOT_IMPLEMENTED,
            handler_used="_handle_not_implemented",
            processing_time=time.time() - start_time
        )

    def route(self, question: str, context: Optional[Dict[str, Any]] = None) -> RouterResponse:
        """
        Routes the user's question to the appropriate handler based on intent.
        """
        start_time = time.time()
        intent = self._classify_intent(question)

        handler_map = {
            Intent.QUESTION_KB: self._handle_knowledge_question,
            Intent.QUESTION_WEB: self._handle_web_question,
            Intent.CHITCHAT: self._handle_chitchat,
            Intent.COMMAND_NOT_IMPLEMENTED: self._handle_not_implemented,
        }

        # Get the handler, default to knowledge base question for UNKNOWN
        handler = handler_map.get(intent, self._handle_knowledge_question)
        
        try:
            return handler(question)
        except Exception as e:
            print(f"IntentRouter: Error in handler for intent {intent}: {e}")
            return RouterResponse(
                answer=f"I encountered an error while processing your request: {e}",
                sources=[],
                confidence=0.0,
                intent=intent,
                handler_used=handler.__name__,
                processing_time=time.time() - start_time
            )

if __name__ == '__main__':
    # This is a simplified test setup and will not work without a real KnowledgeKernel
    print("This is a test script for IntentRouter.")
    print("It cannot be run standalone without a KnowledgeKernel instance.")
    
    # To test this properly, you would need to do something like:
    # from knowledge_kernel_v4_1 import KnowledgeKernelV41
    # kernel = KnowledgeKernelV41()
    # router = IntentRouter(kernel)
    # response = router.route("What is the capital of France?")
    # print(response)