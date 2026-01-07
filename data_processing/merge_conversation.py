#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to merge multiple conversation JSON files.

Usage:
    python merge_conversations.py --input file1.json file2.json --output merged.json
    python merge_conversations.py -i conversations1.json conversations2.json -o all_conversations.json
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_json_file(file_path: str) -> List[Dict[str, Any]]:
    """Load JSON file and return list of conversations."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle both list and dict formats
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            # If it's a dict with a 'conversations' key
            if 'conversations' in data:
                return data['conversations']
            # If it's a single conversation object, wrap it in a list
            else:
                return [data]
        else:
            logger.warning(f"Unexpected data format in {file_path}")
            return []
            
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {file_path}: {e}")
        return []
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return []
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        return []


def merge_conversations(input_files: List[str]) -> List[Dict[str, Any]]:
    """Merge multiple conversation JSON files."""
    all_conversations = []
    
    for file_path in input_files:
        logger.info(f"Loading: {file_path}")
        conversations = load_json_file(file_path)
        
        if conversations:
            all_conversations.extend(conversations)
            logger.info(f"  ✓ Loaded {len(conversations)} conversations")
        else:
            logger.warning(f"  ✗ No conversations loaded from {file_path}")
    
    return all_conversations


def save_merged_file(conversations: List[Dict[str, Any]], output_path: str) -> None:
    """Save merged conversations to output file."""
    try:
        # Create output directory if it doesn't exist
        output_dir = Path(output_path).parent
        if output_dir != Path('.'):
            output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(conversations, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✓ Saved {len(conversations)} conversations to {output_path}")
        
    except Exception as e:
        logger.error(f"Error saving output file: {e}")
        raise


def get_statistics(conversations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate statistics about the merged conversations."""
    stats = {
        'total_conversations': len(conversations),
        'source_types': {},
        'total_turns': 0,
        'files_processed': set(),
    }
    
    for conv in conversations:
        # Count source types
        source_type = conv.get('source_type', 'unknown')
        stats['source_types'][source_type] = stats['source_types'].get(source_type, 0) + 1
        
        # Count turns
        conversation_turns = conv.get('conversation', [])
        stats['total_turns'] += len(conversation_turns)
        
        # Track unique source files
        source_file = conv.get('source_file')
        if source_file:
            stats['files_processed'].add(source_file)
    
    stats['unique_files'] = len(stats['files_processed'])
    del stats['files_processed']  # Remove the set, keep only count
    
    return stats


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Merge multiple conversation JSON files into one",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python merge_conversations.py --input conv1.json conv2.json --output merged.json
  python merge_conversations.py -i file1.json file2.json file3.json -o all_conversations.json
        """
    )
    
    parser.add_argument(
        '-i', '--input',
        nargs='+',
        required=True,
        help='Input JSON files to merge (space-separated)'
    )
    parser.add_argument(
        '-o', '--output',
        required=True,
        help='Output JSON file path for merged conversations'
    )
    parser.add_argument(
        '-s', '--stats',
        action='store_true',
        help='Show detailed statistics after merging'
    )
    
    args = parser.parse_args()
    
    # Validate input files exist
    for file_path in args.input:
        if not Path(file_path).exists():
            logger.error(f"Input file not found: {file_path}")
            return 1
    
    logger.info("=" * 60)
    logger.info("Starting conversation merge")
    logger.info(f"Input files: {len(args.input)}")
    for i, f in enumerate(args.input, 1):
        logger.info(f"  {i}. {f}")
    logger.info(f"Output: {args.output}")
    logger.info("=" * 60)
    
    # Merge conversations
    merged_conversations = merge_conversations(args.input)
    
    if not merged_conversations:
        logger.error("No conversations to merge!")
        return 1
    
    # Save merged file
    save_merged_file(merged_conversations, args.output)
    
    # Show statistics
    stats = get_statistics(merged_conversations)
    
    logger.info("\n" + "=" * 60)
    logger.info("Merge completed successfully!")
    logger.info(f"Total conversations: {stats['total_conversations']}")
    logger.info(f"Total turns: {stats['total_turns']}")
    logger.info(f"Unique source files: {stats['unique_files']}")
    
    if stats['source_types']:
        logger.info("\nSource type breakdown:")
        for source_type, count in stats['source_types'].items():
            logger.info(f"  - {source_type}: {count}")
    
    if args.stats:
        logger.info("\nDetailed statistics:")
        logger.info(json.dumps(stats, indent=2, ensure_ascii=False))
    
    logger.info("=" * 60)
    
    return 0


if __name__ == "__main__":
    exit(main())