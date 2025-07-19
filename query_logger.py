"""Query logging and analysis module.

This module provides a QueryLogger class that tracks queries, their costs,
and performance metrics for analysis and optimization purposes.
"""

import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime


class QueryLogger:
    """Logs queries and their costs for analysis."""

    def __init__(self, log_file: Optional[str] = None):
        self.log_file = log_file or "logs/query_log.jsonl"
        self.queries: List[Dict[str, Any]] = []

        # Ensure log directory exists
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)

        # Load existing logs
        self._load_existing_logs()

    def _load_existing_logs(self):
        """Load existing query logs from file."""
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            self.queries.append(json.loads(line))
            except (OSError, ValueError, json.JSONDecodeError) as e:
                print(f"Warning: Could not load existing query log: {e}")

    def log_query(self, query: str, answer: str, cost: float,
                  elapsed_time: float, metadata: Optional[Dict[str, Any]] = None):
        """Log a query with its results and metrics."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'answer': answer,
            'cost': cost,
            'elapsed_time': elapsed_time,
            'metadata': metadata or {}
        }

        self.queries.append(log_entry)

        # Write to file
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry) + '\n')
        except (OSError, ValueError) as e:
            print(f"Warning: Could not write to query log: {e}")

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of all logged queries."""
        if not self.queries:
            return {
                'total_queries': 0,
                'total_cost': 0.0,
                'average_cost': 0.0,
                'total_time': 0.0,
                'average_time': 0.0
            }

        total_queries = len(self.queries)
        total_cost = sum(q['cost'] for q in self.queries)
        total_time = sum(q['elapsed_time'] for q in self.queries)

        return {
            'total_queries': total_queries,
            'total_cost': total_cost,
            'average_cost': total_cost / total_queries,
            'total_time': total_time,
            'average_time': total_time / total_queries
        }

    def get_recent_queries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get the most recent queries."""
        return self.queries[-limit:]

    def get_queries_by_cost_range(self, min_cost: float = 0,
                                 max_cost: float = float('inf')) -> List[Dict[str, Any]]:
        """Get queries within a cost range."""
        return [
            q for q in self.queries
            if min_cost <= q['cost'] <= max_cost
        ]

    def get_queries_by_time_range(self, min_time: float = 0,
                                 max_time: float = float('inf')) -> List[Dict[str, Any]]:
        """Get queries within a time range."""
        return [
            q for q in self.queries
            if min_time <= q['elapsed_time'] <= max_time
        ]

    def export_to_json(self, filename: str):
        """Export all queries to a JSON file."""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.queries, f, indent=2)
        except (OSError, ValueError) as e:
            print(f"Error exporting queries: {e}")

    def clear_logs(self):
        """Clear all logged queries."""
        self.queries.clear()
        try:
            with open(self.log_file, 'w', encoding='utf-8'):
                pass  # Create empty file
        except OSError as e:
            print(f"Error clearing log file: {e}")
