# Document Agent Hierarchy Discussion Summary

## Current Status

### Background
We've been discussing the architecture for handling different types of documents in the second-brain system, particularly around the distinction between related documents (like lecture transcripts in a module) and independent documents (like research papers).

### Key Insights from Discussion

#### 1. Semantic Relationships Matter More Than Document Count
- **Related Documents**: Lecture transcripts within a module, project files within a directory, chapters of a book
  - Logically related, build on each other
  - Should support cross-document synthesis
  - Should understand references between documents
  - Should treat as cohesive whole

- **Independent Documents**: Research papers in a directory, random PDFs in a folder
  - Independent works, not necessarily related
  - Should focus on individual documents
  - Should NOT assume relationships between documents
  - Should NOT automatically synthesize across documents

#### 2. Current Implementation Issues
- **DocumentAgent** and **ModuleAgent** both handle multiple documents
- The distinction between "single" and "multi" document is artificial
- **DocumentAgent** now uses same chunking/search logic as **ModuleAgent**
- Papers loaded with `/load papers` are incorrectly treated as related
- Different document types need different metadata schemas

#### 3. Proposed Architecture

##### Base DocumentAgent
```python
class DocumentAgent:
    """Base class for document processing"""
    def __init__(self, documents, relationship_type="collection"):
        self.documents = documents
        self.relationship_type = relationship_type
        self.metadata_schema = self._determine_metadata_schema()
```

##### ModuleAgent (for related documents)
```python
class ModuleAgent(DocumentAgent):
    """For semantically related documents (lectures, project files)"""
    def __init__(self, documents):
        super().__init__(documents, relationship_type="module")
        self.cross_document_synthesis = True
```

##### CollectionAgent (for independent documents)
```python
class CollectionAgent(DocumentAgent):
    """For independent documents (papers, mixed content)"""
    def __init__(self, documents):
        super().__init__(documents, relationship_type="collection")
        self.cross_document_synthesis = False
```

#### 4. Metadata Differences
- **Paper-specific metadata**: title, authors, read_status, reading_notes, citations, publication_date, journal, doi
- **Transcript-specific metadata**: module, lecture_number, date, instructor, topics, duration

### Proposed Changes to `/load` Command

#### 1. Add Agent Type Option
```bash
/load papers --agent-type=collection  # Independent documents
/load classes/cpsc_8400 --agent-type=module  # Related documents
```

#### 2. Default Behavior Based on Document Count
- **Single document**: Default to individual agent
- **Multiple documents**: Default to module agent (assumes related)

#### 3. Confirmation for Multi-Document Loads
- When loading multiple documents without specifying agent type
- Ask user to confirm if documents should be treated as logically related group ("module mode")
- Provide option to override default behavior

#### 4. Example User Experience
```bash
second-brain> /load papers
Found 5 documents in papers/
Multiple documents detected. Treat as related module? (y/n): n
Loading as independent documents (collection mode)...
✓ Added collection agent: papers (5 docs)

second-brain> /load classes/cpsc_8400/transcripts/Module_01
Found 8 documents in Module_01/
Multiple documents detected. Treat as related module? (y/n): y
Loading as related documents (module mode)...
✓ Added module agent: Module_01 (8 docs)
```

### Implementation Considerations

#### 1. Backward Compatibility
- Existing `/load` commands should continue to work
- Default behavior should match current expectations
- Gradual migration path for users

#### 2. Configuration Options
- Allow users to set default preferences
- Support for different metadata schemas
- Configurable synthesis behavior

#### 3. Error Prevention
- Prevent subtle misbehavior (e.g., synthesizing across unrelated papers)
- Clear distinction between related and independent document handling
- Appropriate warnings and confirmations

### Next Steps
1. Clean up current branch and merge to main
2. Create new branch for document agent hierarchy implementation
3. Implement the proposed architecture changes
4. Update `/load` command with new options
5. Add confirmation dialogs for multi-document loads
6. Test with different document types and scenarios

### Success Criteria
- Clear distinction between related and independent document handling
- No automatic cross-document synthesis for independent documents
- Appropriate metadata handling for different document types
- Intuitive user experience with `/load` command
- Backward compatibility maintained