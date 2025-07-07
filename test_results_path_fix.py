#!/usr/bin/env python3
"""
Test script to verify that results.json files are no longer created in the root directory
when results_path is None.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from agents.agents_manager import AgentsManager
from task import Task
from datetime import datetime
import docker

class MockTask(Task):
    """Mock task for testing"""
    def __init__(self):
        self.repo_name = "test-repo"
        self.commit = "test-commit"
        self.version = "test-version"
        self.test_patch = "test-patch"
        self._project_path = "/tmp/test-project"
    
    @property
    def project_path(self) -> str:
        return self._project_path
    
    @project_path.setter
    def project_path(self, value: str) -> None:
        self._project_path = value
    
    def get_issue_statement(self) -> str:
        return "Test issue statement"
    
    def setup_project(self) -> None:
        pass
    
    def reset_project(self) -> None:
        pass

def test_results_path_fix():
    """Test that results.json is not created in root directory when results_path is None"""
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Testing in temporary directory: {temp_dir}")
        
        # Create a mock task
        task = MockTask()
        
        # Create output directory
        output_dir = os.path.join(temp_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        
        # Test with results_path=None (the problematic case)
        try:
            # This should create results.json in output/results/ instead of root
            agents_manager = AgentsManager(
                task=task,
                output_dir=output_dir,
                client=docker.from_env(),
                start_time=datetime.now(),
                max_iteration_num=1,
                results_path=None,  # This was causing the issue
                disable_memory_pool=False,
                disable_context_retrieval=False,
                disable_run_test=False
            )
            
            # Check if results.json was created in the expected location
            expected_results_file = os.path.join(output_dir, "results", "results.json")
            expected_lock_file = expected_results_file + ".lock"
            
            if os.path.exists(expected_results_file):
                print(f"✅ SUCCESS: results.json created in expected location: {expected_results_file}")
            else:
                print(f"❌ FAILED: results.json not found in expected location: {expected_results_file}")
                return False
                
            if os.path.exists(expected_lock_file):
                print(f"✅ SUCCESS: results.json.lock created in expected location: {expected_lock_file}")
            else:
                print(f"❌ FAILED: results.json.lock not found in expected location: {expected_lock_file}")
                return False
            
            # Check that files are NOT created in the current working directory
            root_results_file = "results.json"
            root_lock_file = "results.json.lock"
            
            if not os.path.exists(root_results_file):
                print(f"✅ SUCCESS: results.json NOT created in root directory")
            else:
                print(f"❌ FAILED: results.json still created in root directory")
                return False
                
            if not os.path.exists(root_lock_file):
                print(f"✅ SUCCESS: results.json.lock NOT created in root directory")
            else:
                print(f"❌ FAILED: results.json.lock still created in root directory")
                return False
            
            print("🎉 All tests passed! The fix is working correctly.")
            return True
            
        except Exception as e:
            print(f"❌ ERROR: Test failed with exception: {e}")
            return False

if __name__ == "__main__":
    success = test_results_path_fix()
    sys.exit(0 if success else 1) 