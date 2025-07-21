# Second Brain Prototype - Code Structure Diagram

## System Overview

The Second Brain Prototype is a multi-agent document synthesis system that uses
LLMs to process and query educational content. The system follows a modular
architecture with clear separation of concerns.

## Architecture Diagram

```text
<!-- markdownlint-disable MD013 -->
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SECOND BRAIN PROTOTYPE                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────┐  │
│  │   Entry Points  │    │  Core System    │    │     Data Layer          │  │
│  │                 │    │                 │    │                         │  │
│  │ • prototype.py  │    │ • ModelConfig   │    │ • DocumentLoader        │  │
│  │ • interactive_  │    │ • PromptManager │    │ • CourseModuleLoader    │  │
│  │   session.py    │    │ • CacheManager  │    │ • QueryLogger           │  │
│  │ • debug_commands│    │ • QueryLogger   │    │ • SemanticCache         │  │
│  └─────────────────┘    └─────────────────┘    └─────────────────────────┘  │
│           │                       │                       │                  │
│           └───────────────────────┼───────────────────────┘                  │
│                                   │                                          │
│  ┌─────────────────────────────────┼─────────────────────────────────────────┐  │
│  │                    AGENT LAYER                                            │  │
│  │                                                                           │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │  │
│  │  │RoutingAgent │  │DocumentAgent│  │ModuleAgent  │  │SynthesisAgent│      │  │
│  │  │             │  │             │  │             │  │             │      │  │
│  │  │• QueryType  │  │• Single doc │  │• Module mgmt│  │• Multi-agent│      │  │
│  │  │• Complexity │  │• Multi-doc  │  │• Chunking   │  │• Synthesis  │      │  │
│  │  │• Routing    │  │• Context    │  │• Search     │  │• Integration│      │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘      │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                                   │                                          │
│  ┌─────────────────────────────────┼─────────────────────────────────────────┐  │
│  │                    UTILITY LAYER                                          │  │
│  │                                                                           │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │  │
│  │  │Summarizer   │  │Evaluator    │  │Embedding    │  │Test Suite   │      │  │
│  │  │             │  │             │  │Router       │  │             │      │  │
│  │  │• Module     │  │• Query      │  │• Semantic   │  │• Unit tests │      │  │
│  │  │  summaries  │  │  evaluation │  │  routing    │  │• Integration│      │  │
│  │  │• Content    │  │• Performance │  │• Vector     │  │• Advanced   │      │  │
│  │  │  analysis   │  │  metrics    │  │  search     │  │  features   │      │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘      │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────┘
<!-- markdownlint-enable MD013 -->
```

## Detailed Component Breakdown

### 1. Entry Points

#### `prototype.py` (Main Entry Point)

- **Purpose**: Main application entry point and orchestration
- **Key Classes**:
  - `SecondBrainPrototype`: Core system class
  - `SimpleCache`: In-memory caching
  - `QueryLogger`: Query logging functionality
- **Functions**:
  - `main()`: Application entry point
  - `interactive_mode()`: Interactive CLI mode
  - `run_test_queries()`: Testing functionality
  - `run_scale_test()`: Performance testing

#### `interactive_session.py` (Interactive CLI)

- **Purpose**: Interactive command-line interface
- **Key Class**: `InteractiveSession` (inherits from `cmd.Cmd`)
- **Features**:
  - Command parsing and execution
  - Session state management
  - Real-time query processing
  - Cost tracking and reporting

#### `debug_commands.py`

- **Purpose**: Debugging utilities and commands
- **Features**: Development and testing helpers

### 2. Core System Components

#### `model_config.py`

- **Purpose**: Model configuration and cost management
- **Key Classes**:
  - `ModelInfo`: Model metadata (cost, context window, provider)
  - `ModelConfig`: Model selection and cost tracking
- **Features**:
  - Support for Anthropic and Google models
  - Cost estimation per agent type
  - Model switching capabilities

#### `prompt_manager.py`

- **Purpose**: System prompt management
- **Key Class**: `PromptManager`
- **Features**:
  - Dynamic prompt loading from markdown files
  - Fallback to default prompts
  - Prompt caching and reloading
  - Custom prompt path configuration

#### `cache_manager.py`

- **Purpose**: Caching system for query results
- **Features**: Simple in-memory caching with statistics

#### `query_logger.py`

- **Purpose**: Query logging and analytics
- **Features**: JSON-based logging with timestamps

### 3. Agent Layer

#### `agents/routing_agent.py`

- **Purpose**: Intelligent query routing
- **Key Classes**:
  - `QueryType`: Enum for query complexity levels
  - `RoutingAgent`: Main routing logic
- **Features**:
  - Query complexity analysis
  - Module relevance determination
  - Cost estimation for routing decisions

#### `agents/document_agent.py`

- **Purpose**: Document-level query processing
- **Key Classes**:
  - `DocumentAgent`: Single/multi-document processing
  - `SynthesisAgent`: Multi-agent response synthesis
- **Features**:
  - Context-aware document querying
  - Cost tracking per query
  - Response synthesis across agents

#### `agents/module_agent.py`

- **Purpose**: Course module management
- **Key Class**: `ModuleAgent`
- **Features**:
  - Semantic document chunking
  - Chunk-based search
  - Module-level query processing

### 4. Data Layer

#### `loaders/document_loader.py`

- **Purpose**: Document loading and processing
- **Key Class**: `DocumentLoader`
- **Features**:
  - Support for PDF, TXT, MD files
  - Recursive directory loading
  - Error handling and reporting

#### `loaders/course_loader.py`

- **Purpose**: Course-specific document organization
- **Key Class**: `CourseModuleLoader`
- **Features**:
  - Module-based document organization
  - Transcript and paper loading
  - Metadata management

### 5. Utility Layer

#### `summarizer.py`

- **Purpose**: Content summarization
- **Features**: Module-level content analysis

#### `evaluation_framework.py`

- **Purpose**: Query evaluation and testing
- **Features**: Performance metrics and evaluation

#### `embedding_router.py`

- **Purpose**: Semantic routing using embeddings
- **Features**: Vector-based query routing

#### `semantic_cache.py`

- **Purpose**: Semantic caching system
- **Features**: Content-aware caching

### 6. Configuration and Data

#### `prompts/` Directory

- `document_agent_single.md`: Single document agent prompts
- `document_agent_multi.md`: Multi-document agent prompts
- `module_agent.md`: Module agent prompts
- `synthesis_agent.md`: Synthesis agent prompts

#### `classes/` Directory

- Course content organized by institution and course
- Structured document hierarchy

## Data Flow

```text
User Query → InteractiveSession → RoutingAgent → [DocumentAgent/ModuleAgent] →
SynthesisAgent → Response
     ↓              ↓                ↓                    ↓                    ↓
QueryLogger → CacheManager → ModelConfig → PromptManager → Cost Tracking
```

## Key Design Patterns

1. **Multi-Agent Architecture**: Specialized agents for different tasks
2. **Strategy Pattern**: Different routing strategies based on query type
3. **Factory Pattern**: Model instantiation based on configuration
4. **Observer Pattern**: Cost tracking and logging
5. **Command Pattern**: Interactive session command handling

## Configuration Management

- Environment variables for API keys
- Model configuration per agent type
- Prompt customization through markdown files
- Cache configuration and management

## Error Handling

- Graceful degradation for missing files
- Fallback prompts when custom prompts unavailable
- Comprehensive logging and error reporting
- Cache invalidation and recovery

## Performance Features

- In-memory caching for query results
- Semantic caching for similar queries
- Cost tracking and optimization
- Parallel processing capabilities
- Query complexity estimation
