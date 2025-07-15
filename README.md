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

## Architecture

The system consists of several key components:

### Document Agents

- Each document has its own specialized agent
- Agents use LLMs to extract relevant information from their assigned document
- Default model: `gemini-1.5-flash` (cost-effective for document processing)

### Synthesis Agent

- Combines responses from multiple document agents
- Creates unified, coherent answers
- Default model: `claude-3-opus-20240229` (high-quality synthesis)

### Cache System

- In-memory caching to reduce costs on repeated queries
- Tracks cache hit rates and performance

### Query Logger

- Logs all queries and responses to JSON lines format
- Enables analysis and debugging

## Setup

### 1. Environment Setup

```bash
# Create and activate conda environment
conda create -n second-brain-proto python=3.11
conda activate second-brain-proto

# Install dependencies
pip install langchain==0.1.0 langchain-anthropic==0.1.1 \
    langchain-google-genai==0.0.5 python-dotenv==1.0.0 \
    pydantic==2.5.3 colorama==0.4.6 tabulate==0.9.0
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

The prototype includes three test documents:

- `documents/cs229_optimization.txt` - Gradient descent and optimization techniques
- `documents/cs221_search.txt` - A* search and heuristic functions
- `documents/paper_transformers.txt` - Attention mechanisms and transformers

## Usage

### Command Line Interface

#### Single Query

```bash
python prototype.py query --question "What optimization techniques are \
discussed across the documents?"
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
├── documents/            # Test documents directory
│   ├── cs229_optimization.txt
│   ├── cs221_search.txt
│   └── paper_transformers.txt
├── logs/                 # Query logs directory
│   └── queries.jsonl     # JSON lines format log file
└── README.md            # This file
```

## Key Features

### Multi-Agent Architecture

- Each document has a specialized agent
- Agents can use different LLM models
- Parallel processing capability

### Intelligent Synthesis

- Combines information from multiple sources
- Identifies connections and patterns
- Maintains source attribution
- Handles contradictions gracefully

### Cost Optimization

- Caching reduces repeated query costs
- Cost tracking and estimation
- Model selection for different tasks

### Monitoring and Logging

- Detailed query logging
- Performance metrics
- Cost analysis
- Cache statistics

## Success Criteria

The prototype demonstrates:

- ✅ Cross-document synthesis
- ✅ Costs under $0.50 per query
- ✅ Response times under 10 seconds
- ✅ Cache functionality
- ✅ Comprehensive logging

## Future Enhancements

Potential improvements for future versions:

1. Semantic similarity checking for cache
2. Parallel document agent queries
3. Web interface using Gradio
4. Export results to markdown reports
5. Support for more document formats
6. Advanced caching strategies

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