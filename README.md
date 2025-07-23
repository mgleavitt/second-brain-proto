# Second Brain Prototype

A minimal prototype demonstrating a multi-agent architecture for synthesizing
information across multiple documents using Large Language Models (LLMs).

## Overview

This prototype validates the concept of a "Second Brain" system that can:

- Process multiple documents through specialized agents
- Synthesize information across documents using LLMs
- Provide comprehensive answers with source attribution
- Track costs and performance metrics
- Cache results for efficiency
- Use intelligent routing to optimize costs
- Generate module summaries for improved accuracy
- Implement semantic caching for similar queries
- Support hybrid chat architecture for cost optimization

## Architecture

The system consists of several key components:

### Document Agents

- Each document has its own specialized agent
- Agents use LLMs to extract relevant information from their assigned document
- Default model: `gemini-1.5-flash` (cost-effective for document processing)

### Module Agents

- Handle collections of related documents (e.g., course modules)
- Provide module-level querying and summarization
- Enable efficient processing of large document corpora

### Synthesis Agent

- Combines responses from multiple document agents
- Creates unified, coherent answers
- Default model: `claude-3-opus-20240229` (high-quality synthesis)

### Routing Agent

- Intelligently routes queries to relevant modules
- Uses keyword matching and semantic embeddings
- Reduces costs by querying only relevant documents
- Implements query complexity detection for optimal routing

### Hybrid Chat System

- Intelligently routes between lightweight chat and full agent pipeline
- Reduces costs for simple follow-up questions
- Maintains conversation continuity
- Uses complexity analysis for routing decisions

### Cache System

- In-memory caching to reduce costs on repeated queries
- Semantic caching using embeddings for similar queries
- Tracks cache hit rates and performance

### Query Logger

- Logs all queries and responses to JSON lines format
- Enables analysis and debugging

## Advanced Features

### Module Summaries

- Automatically generates summaries for each module
- Extracts key topics and concepts
- Improves routing accuracy by 30-50%
- Cached to disk for efficiency

### Semantic Caching

- Uses sentence embeddings to find similar queries
- Reduces API calls by 30-50% for related questions
- Configurable similarity threshold
- Automatic expiration of old entries

### Embedding-based Routing

- More accurate than keyword matching
- Uses semantic similarity to route queries
- Explains routing decisions
- Indexes module summaries for fast lookup

### Query Evaluation Framework

- Evaluates response quality using LLMs
- Compares different routing strategies
- Provides cost/quality tradeoff analysis
- Generates performance reports

### Hybrid Chat Features

- Routes simple queries to lightweight models
- Uses full agent pipeline for complex queries
- Implements context filtering for efficiency
- Provides cost optimization through intelligent routing

## Reviewing Logs

The system logs all queries to `logs/queries.jsonl` in JSON Lines format.
You can review these logs using various command-line tools:

### Using `jq` (Recommended)

If you have `jq` installed, you can easily analyze the logs:

```bash
# View all entries with formatting
cat logs/queries.jsonl | jq '.'

# View just the questions
cat logs/queries.jsonl | jq '.question'

# View questions with timestamps
cat logs/queries.jsonl | \
  jq '{question: .question, timestamp: .timestamp}'

# Count total entries
cat logs/queries.jsonl | jq -s 'length'

# View cost summary
cat logs/queries.jsonl | jq -s 'map(.result.total_cost) | add'

# Find queries containing specific terms
cat logs/queries.jsonl | jq 'select(.question | contains("optimization"))'

# View response times
cat logs/queries.jsonl | jq '{question: .question, duration: .result.duration}'

# Export to CSV format
cat logs/queries.jsonl | \
  jq -r '. | [.question, .result.total_cost, .result.duration, .timestamp] | @csv'
```

### Alternative Methods

If `jq` is not available:

```bash
# View questions using grep
grep -o '"question": "[^"]*"' logs/queries.jsonl

# View timestamps
grep -o '"timestamp": "[^"]*"' logs/queries.jsonl

# Count entries
wc -l logs/queries.jsonl

# Python one-liner for basic analysis
python3 -c "import json; data=[json.loads(line) for line in open('logs/queries.jsonl')]; \
  print(f'Total queries: {len(data)}'); \
  print(f'Total cost: ${sum(q[\"result\"][\"total_cost\"] for q in data):.4f}')"
```

## Setup

### 1. Environment Setup

```bash
# Create and activate conda environment
conda create -n second-brain-proto python=3.11
conda activate second-brain-proto

# Install dependencies
pip install -r requirements.txt
```

### 2. API Keys

Create a `.env` file based on `.env.example`:

```bash
cp .env.example .env
```

Add your API keys:

```env
ANTHROPIC_API_KEY=your_anthropic_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
```

### 3. Test Documents

The prototype includes test documents in the `documents/` directory and extensive course
materials in the `classes/` directory.

## Usage

### Command Line Interface

The prototype supports several commands and options:

#### Commands

- **`query`** - Ask a single question
- **`test`** - Run predefined test queries
- **`costs`** - Display cost summary statistics
- **`clear-cache`** - Clear the query cache
- **`interactive`** - Start interactive mode for multiple queries
- **`scale-test`** - Run comprehensive scale testing with routing

#### Options

- **`--question`** or **`-q`** - Question to ask (required for `query` command)
- **`--documents`** or **`-d`** - Documents or directories to load (can specify multiple)
- **`--recursive`** or **`-r`** - Recursively search directories for documents
- **`--no-cache`** - Disable query result caching
- **`--use-routing`** - Enable intelligent query routing

#### Single Query

```bash
python prototype.py query --question "What optimization techniques are \
discussed across the documents?"
```

#### Query with Custom Documents

```bash
python prototype.py query --question "Explain databases" \
  --documents /path/to/db_course/ --recursive
```

#### Query with Intelligent Routing

```bash
python prototype.py query --question "What is machine learning?" --use-routing
```

#### Query with No Cache

```bash
python prototype.py query --question "..." --no-cache
```

#### Run All Test Queries

```bash
python prototype.py test
```

#### Show Cost Summary

```bash
python prototype.py costs
```

#### Clear Cache

```bash
python prototype.py clear-cache
```

#### Interactive Mode

```bash
python prototype.py interactive
```

#### Scale Testing

```bash
python prototype.py scale-test --use-routing
```

## System Prompts

The system supports file-based prompts for all agents, allowing dynamic customization
of agent behavior without code changes.

### Default Prompt Files

The system uses the following default prompt files in the `prompts/` directory:

- `prompts/document_agent_single.md` - For single document queries
- `prompts/document_agent_multi.md` - For multi-document queries
- `prompts/module_agent.md` - For module-level coordination
- `prompts/synthesis_agent.md` - For response synthesis

### CLI Usage

#### Custom Prompt Directory

```bash
# Use custom prompt directory
python prototype.py query -q "question" --prompt-dir /path/to/prompts
```

#### Custom Prompt Files

```bash
# Use specific prompt files
python prototype.py query -q "question" \
  --document-single-prompt /path/to/custom_single.md \
  --synthesis-prompt /path/to/custom_synthesis.md
```

#### All Prompt Options

```bash
python prototype.py query -q "question" \
  --prompt-dir /path/to/prompts \
  --document-single-prompt /path/to/custom_single.md \
  --document-multi-prompt /path/to/custom_multi.md \
  --module-prompt /path/to/custom_module.md \
  --synthesis-prompt /path/to/custom_synthesis.md
```

### Interactive Mode Commands

In interactive mode, you can manage prompts dynamically:

```bash
# Start interactive mode
python prototype.py interactive

# Available prompt commands:
/prompt-reload              # Reload all prompts from files
/prompt-set module /path/to/custom.md  # Set custom prompt
/prompt-show synthesis      # Display current prompt
/prompt-create-defaults     # Create default prompt files
/prompt-help                # Show prompt help

# Advanced feature commands:
/summarize [--force]        # Generate module summaries
/evaluate <question>        # Evaluate query strategies
/semantic-cache [stats|clear-expired]  # Manage semantic cache
/use-embeddings [on|off]    # Toggle embedding-based routing
```

### Prompt File Format

Prompt files use markdown format with the following structure:

```markdown
# Document Single System Prompt

## Overview
This prompt defines the behavior for the document_single agent.

## Instructions
You are a document analysis agent. Your task is to answer questions based on the provided document content. Be thorough and accurate.

## Guidelines
- Be accurate and thorough
- Cite sources when applicable
- Maintain objectivity

---
*Generated on 2025-07-17 17:16:49*
```

### Testing Prompts

Run the prompt test suite to verify functionality:

```bash
python test_prompts.py
```

This will test:

- Basic prompt loading and caching
- Custom prompt directories
- Custom prompt file paths
- Prompt reloading
- Fallback to defaults
- Default file creation

### Testing Advanced Features

Test the new advanced optimization features:

```bash
python test_advanced_features.py
```

This will test:

- Module summarization
- Semantic caching
- Embedding-based routing
- Query evaluation framework

### Example Output

```text
==================================================
=== Query ===
What optimization techniques are discussed across the documents?

=== Response ===
Based on the analysis of the three documents, several optimization techniques
are discussed:

From the CS229 document, the primary optimization technique is gradient
descent, which includes:
- Batch Gradient Descent: Uses entire training set for gradient computation
- Stochastic Gradient Descent (SGD): Uses single training examples
- Mini-batch Gradient Descent: Compromise between batch and stochastic
  approaches

The CS221 document discusses A* search as an optimization algorithm for
pathfinding, which uses heuristic functions to guide the search process
efficiently.

The transformers paper discusses attention mechanisms and their computational
optimization challenges, including techniques like sparse attention patterns
and linear attention mechanisms to address quadratic complexity.

=== Sources ===
- CS229_Agent
- CS221_Agent
- Paper_Agent

=== Metrics ===
Total Cost: $0.0452
Total Tokens: 2,543
Response Time: 4.2s
Cache Hit: No
==================================================
```

## Cost Tracking

The system tracks costs across different models:

| Model | Input Cost | Output Cost |
|-------|------------|-------------|
| gemini-1.5-flash | $0.075/1M tokens | $0.30/1M tokens |
| claude-3-opus-20240229 | $15/1M tokens | $75/1M tokens |

Typical query costs range from $0.02 to $0.10 depending on document length and
response complexity.

## Project Structure

```text
second-brain-proto/
├── .env                    # API keys (create from .env.example)
├── .env.example           # Template for environment variables
├── prototype.py           # Main prototype script
├── interactive_session.py # Interactive CLI interface
├── chat_controller.py     # Hybrid chat routing controller
├── complexity_analyzer.py # Query complexity analysis
├── context_filter.py      # Context filtering for agents
├── light_chat_model.py    # Lightweight chat model
├── hybrid_config.py       # Hybrid chat configuration
├── prompt_manager.py      # System prompt management
├── model_config.py        # Model configuration and cost tracking
├── cache_manager.py       # Caching system
├── query_logger.py        # Query logging functionality
├── summarizer.py          # Module summarization
├── semantic_cache.py      # Semantic caching system
├── embedding_router.py    # Embedding-based routing
├── evaluation_framework.py # Query evaluation framework
├── conversation_manager.py # Conversation management
├── test_prompts.py        # Prompt functionality tests
├── test_advanced_features.py # Advanced features tests
├── debug_commands.py      # Debugging utilities
├── agents/                # Agent implementations
│   ├── __init__.py
│   ├── base_agent.py      # Base agent class
│   ├── document_agent.py  # Individual document agents
│   ├── module_agent.py    # Module-based agents
│   └── routing_agent.py   # Intelligent query routing
├── loaders/               # Document loading utilities
│   ├── __init__.py
│   ├── document_loader.py # General document loader
│   └── course_loader.py   # Course/module loader
├── prompts/               # System prompt files
│   ├── document_agent_single.md
│   ├── document_agent_multi.md
│   ├── module_agent.md
│   └── synthesis_agent.md
├── documents/            # Test documents directory
├── classes/              # Course materials directory
├── chats/                # Chat conversation data
├── logs/                 # Query logs directory
│   └── queries.jsonl     # JSON lines format log file
├── .cache/               # Cache directory
├── .semantic_cache/      # Semantic cache directory
├── .evaluation/          # Evaluation results directory
├── .conversations/       # Conversation history
└── README.md            # This file
```

## Key Features

### Multi-Agent Architecture

- Each document has a specialized agent
- Module-based agents for handling related document collections
- Agents can use different LLM models
- Parallel processing capability

### Intelligent Routing

- Analyzes query complexity and type
- Routes simple queries to single agents for cost efficiency
- Routes module-specific queries to relevant module agents
- Uses full synthesis pipeline for complex cross-document queries
- Reduces costs by 40-60% for simple queries

### Hybrid Chat Implementation

- Routes between lightweight and full agent pipelines
- Maintains conversation continuity
- Implements context filtering for efficiency
- Provides significant cost savings for follow-up questions

### Intelligent Synthesis

- Combines information from multiple sources
- Identifies connections and patterns
- Maintains source attribution
- Handles contradictions gracefully

### Cost Optimization

- Caching reduces repeated query costs
- Intelligent routing minimizes unnecessary agent queries
- Cost tracking and estimation
- Model selection for different tasks

### Monitoring and Logging

- Detailed query logging with routing decisions
- Performance metrics by query type
- Cost analysis and breakdown
- Cache statistics and hit rates

## Success Criteria

The prototype demonstrates:

- ✅ Cross-document synthesis
- ✅ Costs under $0.50 per query
- ✅ Response times under 10 seconds
- ✅ Cache functionality
- ✅ Comprehensive logging
- ✅ Intelligent routing
- ✅ Module summarization
- ✅ Semantic caching
- ✅ Query evaluation
- ✅ Hybrid chat architecture

## Future Enhancements

Potential improvements for future versions:

1. **Already Implemented** ✅
   - Intelligent query routing
   - Module-based document organization
   - Comprehensive cost tracking
   - Scale testing capabilities
   - Advanced optimization features
   - Hybrid chat architecture

2. **Next Steps**
   - Web interface using Gradio
   - Export results to markdown reports
   - Support for more document formats (PDF, DOCX)
   - Advanced caching strategies with TTL
   - Query performance optimization
   - Real-time document updates
   - Knowledge graph construction

## Troubleshooting

### Common Issues

1. **API Key Errors**
   - Ensure `.env` file exists with correct API keys
   - Verify API keys are valid and have sufficient credits

2. **Document Loading Errors**
   - Check that document files exist in `documents/` directory
   - Ensure files are UTF-8 encoded

3. **Model Errors**
   - Verify model names are correct
   - Check API provider status

4. **Embedding Model Issues**
   - First run may take time to download embedding models
   - Ensure sufficient disk space for model caching

### Debug Mode

Enable debug logging by modifying the script to add more verbose output.

## Contributing

This is a prototype for validation purposes. For production use, consider:

- Adding proper error handling
- Implementing security measures
- Adding unit tests
- Optimizing for scale

## License

This project is for educational and research purposes.

## Further Technical Documentation

For advanced users and contributors, the following markdown files provide in-depth technical details, architectural diagrams, and the history of major optimizations:

- **DOCUMENT_AGENT_HIERARCHY_DISCUSSION.md**: Important architectural discussion
  about document agent hierarchy and relationship types.
- **SYSTEM_IMPROVEMENTS.md**: Comprehensive overview of all major improvements
  and advanced features implemented in the system, including module summaries,
  semantic caching, embedding-based routing, routing fixes, and performance
  optimizations.
- **ARCHITECTURE.md**: Complete system architecture documentation including
  component diagrams, data flow, configuration management, and development
  guidelines.
- **TECHNICAL_GUIDE.md**: Comprehensive technical implementation guide covering
  development setup, core architecture, hybrid chat system, advanced features,
  testing, and deployment considerations.

Consult these files for deeper insight into the system's design, optimization
strategies, and implementation rationale.
