# Chatbot Project Instructions

## Folder Responsibilities

- `server/` - API layer. All HTTP endpoints live here.
- `server/routers/` - Route definitions only. No business logic.
- `server/repository/` - Business logic. Routers call repositories.
- `chatbot/` - Core chatbot logic. Framework-agnostic.
- `chatbot/graph/` - LangGraph definition. Nodes in `nodes/` subfolder.
- `chatbot/graph/nodes/` - Individual graph node functions.
- `chatbot/tools/` - LLM tools. One file per tool or tool group.
- `chatbot/api/` - External API clients (Oracle Hospitality, etc.)
- `chatbot/modes/` - Chat mode handlers (text, voice, realtime).
- `chatbot/nodes/` - Audio processing nodes (STT, TTS) - legacy.

## Constraints

### API Layer
- Routers handle HTTP only - validation, auth, response formatting
- Business logic goes in repository layer
- Each router file = one resource
- Use Pydantic models for request/response validation

### Graph Layer
- Each node is a single-responsibility function
- Nodes live in `chatbot/graph/nodes/`
- State definition lives in `chatbot/graph/state.py`
- Graph construction lives in `chatbot/graph/builder.py`
- Nodes receive state dict and return partial state updates

### Tools
- Use `@tools_registry.register` decorator
- Tools auto-register on import
- Tool files imported in `chatbot/tools/__init__.py`
- Tool implementations can use `chatbot/api/` clients

### Modes
- Each mode (text, voice, realtime) has its own handler
- Handlers inherit from `BaseChatHandler`
- Modes use the unified graph for processing

## Conventions

- snake_case for files and functions
- PascalCase for classes
- Type hints required on all public functions
- Docstrings required on all public functions
- Use `Optional[]` for nullable parameters
- Use `Literal[]` for string enums in state

## Discovery

Where to look to understand key parts:

- Graph structure: `chatbot/graph/builder.py`
- Graph state schema: `chatbot/graph/state.py`
- API structure: `server/main.py`
- Available tools: `chatbot/tools/__init__.py`
- LLM providers: `chatbot/llm_provider.py`
- Chat handlers: `chatbot/modes/`

## Adding New Features

### New API Endpoint
1. Create router in `server/routers/`
2. Create repository in `server/repository/` if business logic needed
3. Register router in `server/main.py`

### New Tool
1. Create file in `chatbot/tools/` or appropriate subfolder
2. Use `@tools_registry.register` decorator
3. Import in `chatbot/tools/__init__.py`

### New Graph Node
1. Create node function in `chatbot/graph/nodes/`
2. Export in `chatbot/graph/nodes/__init__.py`
3. Add to graph in `chatbot/graph/builder.py`
4. Update edges as needed

### New External API Client
1. Create client in `chatbot/api/`
2. Use existing patterns from `chatbot/api/oracle_hospitality/`
3. Wrap in tools if needed

### New Chat Mode
1. Create handler in `chatbot/modes/`
2. Inherit from `BaseChatHandler`
3. Implement required abstract methods
4. Export in `chatbot/modes/__init__.py`

## Running

- **API Server**: `uvicorn server.main:app --reload`
- **Streamlit App**: `streamlit run app.py`
- **Realtime Voice**: `streamlit run realtime_voice.py`

## Key Architectural Decisions

1. **Unified Graph**: Single LangGraph handles both text and audio inputs via conditional routing
2. **Router/Repository Pattern**: API layer separated from business logic
3. **Tool Registry**: Centralized tool management with decorator-based registration
4. **Mode Handlers**: High-level abstractions for different interaction modes
5. **Realtime Separate**: WebSocket-based realtime voice stays separate from request/response graph
