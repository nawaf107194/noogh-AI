# Phase 7 Implementation Plan: Government Core Migration

## Problem

The Government System (`President`, Ministers) exists but is not exposed via the modern API architecture. It currently:

- Creates `KnowledgeKernelV41` internally (should be injected/managed)
- Initializes ministers directly (should be configurable)
- Is not accessible via REST API
- Lives in isolation from the new User Domain patterns

## User Review Required

> [!IMPORTANT]
> **Dependency Management Decision**
> The `President` class currently creates its own `KnowledgeKernelV41` instance. We have two options:
>
> 1. **Singleton Pattern** - Create one President instance at startup, shared across all requests (simpler, maintains state)
> 2. **Per-Request Pattern** - Create new President for each request (stateless, more scalable)
>
> **Recommendation:** Singleton pattern since the President maintains learning/memory across interactions.

## Proposed Changes

### Core Package

#### [MODIFY] [president.py](file:///home/noogh/projects/noogh_unified_system/src/government/president.py)

**Current State:**

- Already uses `async def process_request`
- Uses relative imports (`from ..nlp.intent import IntentRouter`)
- Creates `KnowledgeKernelV41` in `__init__`

**Changes:**

- Keep as-is (already modern!)
- Possibly add configuration via settings for verbose mode

---

### Services Layer

#### [NEW] [government_service.py](file:///home/noogh/projects/noogh_unified_system/src/services/government_service.py)

**Purpose:** Service layer wrapping President with singleton management

```python
class GovernmentService:
    _president_instance: Optional[President] = None
    
    @classmethod
    def get_president(cls) -> President:
        """Get singleton President instance"""
        if cls._president_instance is None:
            cls._president_instance = President(verbose=True)
        return cls._president_instance
    
    async def process_message(self, message: str, context: dict = None):
        """Process user message through government system"""
        president = self.get_president()
        return await president.process_request(message, context)
```

---

### Schemas Layer

#### [NEW] [government.py](file:///home/noogh/projects/noogh_unified_system/src/schemas/government.py)

**DTOs for Government API:**

```python
class GovernmentChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    context: Optional[Dict[str, Any]] = None
    priority: str = Field(default="medium")

class GovernmentChatResponse(BaseModel):
    success: bool
    response: str
    minister: Optional[str]
    intent: Optional[str]
    task_id: Optional[str]
```

---

### API Routes

#### [NEW] [government.py](file:///home/noogh/projects/noogh_unified_system/src/api/routes/government.py)

**Endpoints:**

- `POST /api/v1/government/chat` - Send message to President
- `GET /api/v1/government/status` - Get cabinet status

```python
@router.post("/chat", response_model=GovernmentChatResponse)
async def chat_with_government(request: GovernmentChatRequest):
    service = GovernmentService()
    result = await service.process_message(
        request.message,
        request.context,
        request.priority
    )
    return result
```

---

### App Configuration

#### [MODIFY] [app.py](file:///home/noogh/projects/noogh_unified_system/src/api/app.py)

Add government router to the routers dict:

```python
'government_v2': ('src.api.routes.government', '/api/v1/government', ['🏛️ Government']),
```

## Verification Plan

### Automated Tests

1. **Unit Test: President Process Request**

   ```bash
   # Create tests/unit/test_president.py
   pytest tests/unit/test_president.py -v
   ```

   - Test async process_request method
   - Test intent routing
   - Test memory recall/learn

2. **Integration Test: Government API**

   ```bash
   # Create tests/integration/test_government_api.py
   pytest tests/integration/test_government_api.py -v
   ```

   - Test POST /api/v1/government/chat endpoint
   - Test GET /api/v1/government/status endpoint
   - Verify response structure

### Manual Verification

1. **Start Server**

   ```bash
   python -m src.api.main
   ```

2. **Test Chat Endpoint**

   ```bash
   curl -X POST "http://localhost:8000/api/v1/government/chat" \
     -H "Content-Type: application/json" \
     -d '{"message": "What is the system status?", "priority": "high"}'
   ```

   **Expected:** JSON response with success=true and response from minister

3. **Test Status Endpoint**

   ```bash
   curl "http://localhost:8000/api/v1/government/status"
   ```

   **Expected:** Cabinet status with total_ministers, success_rate, etc.

4. **API Documentation**
   - Open `http://localhost:8000/docs`
   - Verify "Government" section appears
   - Test interactive endpoints in Swagger UI

---

**Next Steps After Approval:**

1. Create schemas
2. Create service layer
3. Create API routes
4. Write tests
5. Wire to main app
6. Verify functionality
