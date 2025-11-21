from typing import Dict, Any, Type, Callable, Optional, List
import logging

logger = logging.getLogger(__name__)

class Container:
    """
    Enhanced Dependency Injection Container
    
    Features:
    - Singleton service registration
    - Factory-based lazy initialization
    - Service existence checking
    - Service introspection
    - Lifecycle hooks
    """
    _services: Dict[str, Any] = {}
    _factories: Dict[str, Callable] = {}
    _on_register_hooks: List[Callable] = []
    _on_resolve_hooks: List[Callable] = []

    @classmethod
    def register(cls, key: str, instance: Any, silent: bool = False):
        """
        Register a singleton instance
        
        Args:
            key: Service identifier
            instance: Service instance
            silent: If True, don't log registration
        """
        cls._services[key] = instance
        
        # Call lifecycle hooks
        for hook in cls._on_register_hooks:
            try:
                hook(key, instance)
            except Exception as e:
                logger.error(f"Error in on_register hook: {e}")
        
        if not silent:
            logger.info(f"📦 Registered service: {key}")

    @classmethod
    def register_factory(cls, key: str, factory: Callable, silent: bool = False):
        """
        Register a factory function for lazy initialization
        
        Args:
            key: Service identifier
            factory: Factory function that creates the service
            silent: If True, don't log registration
        """
        cls._factories[key] = factory
        if not silent:
            logger.info(f"🏭 Registered factory: {key}")

    @classmethod
    def register_lazy(cls, key: str, factory: Callable):
        """
        Alias for register_factory for better readability
        
        Args:
            key: Service identifier
            factory: Factory function that creates the service
        """
        cls.register_factory(key, factory)

    @classmethod
    def resolve(cls, key: str, default: Any = None) -> Optional[Any]:
        """
        Resolve a dependency
        
        Args:
            key: Service identifier
            default: Default value if service not found
            
        Returns:
            Service instance or default value
        """
        # Check if already instantiated
        if key in cls._services:
            instance = cls._services[key]
            
            # Call lifecycle hooks
            for hook in cls._on_resolve_hooks:
                try:
                    hook(key, instance)
                except Exception as e:
                    logger.error(f"Error in on_resolve hook: {e}")
            
            return instance
        
        # Check if factory exists
        if key in cls._factories:
            instance = cls._factories[key]()
            cls._services[key] = instance  # Cache as singleton
            
            # Call lifecycle hooks
            for hook in cls._on_resolve_hooks:
                try:
                    hook(key, instance)
                except Exception as e:
                    logger.error(f"Error in on_resolve hook: {e}")
            
            logger.info(f"🔧 Instantiated service from factory: {key}")
            return instance
        
        # Service not found
        if default is not None:
            return default
            
        logger.warning(f"⚠️ Service not found: {key}")
        return None

    @classmethod
    def has(cls, key: str) -> bool:
        """
        Check if a service is registered
        
        Args:
            key: Service identifier
            
        Returns:
            True if service exists (either instantiated or has factory)
        """
        return key in cls._services or key in cls._factories

    @classmethod
    def get_all(cls) -> Dict[str, Any]:
        """
        Get all registered services (instantiated only)
        
        Returns:
            Dictionary of all instantiated services
        """
        return cls._services.copy()

    @classmethod
    def get_all_keys(cls) -> List[str]:
        """
        Get all service keys (both instantiated and factories)
        
        Returns:
            List of all service identifiers
        """
        return list(set(cls._services.keys()) | set(cls._factories.keys()))

    @classmethod
    def add_on_register_hook(cls, hook: Callable[[str, Any], None]):
        """
        Add a lifecycle hook that's called when a service is registered
        
        Args:
            hook: Function that takes (key, instance) as arguments
        """
        cls._on_register_hooks.append(hook)

    @classmethod
    def add_on_resolve_hook(cls, hook: Callable[[str, Any], None]):
        """
        Add a lifecycle hook that's called when a service is resolved
        
        Args:
            hook: Function that takes (key, instance) as arguments
        """
        cls._on_resolve_hooks.append(hook)

    @classmethod
    def clear(cls):
        """Clear all services, factories, and hooks"""
        cls._services.clear()
        cls._factories.clear()
        cls._on_register_hooks.clear()
        cls._on_resolve_hooks.clear()
        logger.info("🧹 Cleared DI container")

    @classmethod
    def stats(cls) -> Dict[str, Any]:
        """
        Get container statistics
        
        Returns:
            Dictionary with container stats
        """
        return {
            "instantiated_services": len(cls._services),
            "registered_factories": len(cls._factories),
            "total_services": len(set(cls._services.keys()) | set(cls._factories.keys())),
            "on_register_hooks": len(cls._on_register_hooks),
            "on_resolve_hooks": len(cls._on_resolve_hooks),
            "service_keys": cls.get_all_keys()
        }
