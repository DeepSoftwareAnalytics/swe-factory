#!/usr/bin/env python3
"""
Debug script to help identify where files are being written during execution.
This script will check for files in the current directory and compare with expected output directory.
"""

import os
import sys
import time
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_files_in_directory(directory):
    """Get all files in a directory recursively."""
    files = []
    for root, dirs, filenames in os.walk(directory):
        for filename in filenames:
            file_path = os.path.join(root, filename)
            files.append(file_path)
    return files

def monitor_file_creation(output_dir, check_interval=5, duration=300):
    """
    Monitor file creation by periodically checking for new files.
    
    Args:
        output_dir: The expected output directory
        check_interval: How often to check for new files (in seconds)
        duration: How long to monitor (in seconds)
    """
    output_dir = os.path.abspath(output_dir)
    current_dir = os.getcwd()
    
    logger.info(f"Starting file creation monitoring for {duration} seconds...")
    logger.info(f"Expected output directory: {output_dir}")
    logger.info(f"Current working directory: {current_dir}")
    
    # Get initial list of files
    initial_files = set()
    if os.path.exists(output_dir):
        initial_files = set(get_files_in_directory(output_dir))
    
    # Also check current directory for any files that might be created there
    current_dir_files = set()
    if os.path.exists(current_dir):
        current_dir_files = set(get_files_in_directory(current_dir))
    
    start_time = time.time()
    check_count = 0
    
    while time.time() - start_time < duration:
        check_count += 1
        logger.info(f"Check #{check_count} - Time elapsed: {time.time() - start_time:.1f}s")
        
        # Check for new files in output directory
        if os.path.exists(output_dir):
            current_files = set(get_files_in_directory(output_dir))
            new_files = current_files - initial_files
            if new_files:
                logger.info(f"New files in output directory ({len(new_files)}):")
                for file_path in sorted(new_files):
                    logger.info(f"  + {file_path}")
                initial_files = current_files
        
        # Check for new files in current directory
        if os.path.exists(current_dir):
            current_files = set(get_files_in_directory(current_dir))
            new_files = current_files - current_dir_files
            if new_files:
                logger.warning(f"WARNING: New files in current directory ({len(new_files)}):")
                for file_path in sorted(new_files):
                    logger.warning(f"  ! {file_path}")
                    if not file_path.startswith(output_dir):
                        logger.error(f"ERROR: File created outside expected output directory!")
                        logger.error(f"  File: {file_path}")
                        logger.error(f"  Expected: {output_dir}")
                current_dir_files = current_files
        
        time.sleep(check_interval)
    
    logger.info("File creation monitoring completed.")
    logger.info(f"Total checks performed: {check_count}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_file_locations.py <output_directory> [duration_seconds] [check_interval_seconds]")
        sys.exit(1)
    
    output_dir = sys.argv[1]
    duration = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    check_interval = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    
    monitor_file_creation(output_dir, check_interval, duration) 