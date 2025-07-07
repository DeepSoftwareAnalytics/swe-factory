# File Location Debug Guide

## Issue Description

When running the SWE Factory tool, files are being created at the repository root instead of in the specified output directories, despite providing the correct `--output-dir`, `--setup-dir`, and `--results-path` parameters.

## Root Cause Analysis

After analyzing the codebase, I found that the system is correctly designed to use the specified output directories. All file writing operations use proper path construction with `pjoin()` or `os.path.join()` to ensure files are written to the correct output directories.

However, there are several potential causes for files appearing in the wrong location:

### 1. Working Directory Changes
The code uses `cd` context managers in several places (like in `dump_cost` function), which temporarily change the working directory. If any file operations happen outside of these context managers while the directory is changed, they might write to the wrong location.

### 2. Race Conditions
The system uses multiprocessing, and there might be race conditions where the working directory is changed in one process while another process is writing files.

### 3. Missing Absolute Paths
Some file operations might not be using absolute paths, causing them to write relative to the current working directory.

## Changes Made

I've made the following improvements to ensure files are written to the correct locations:

### 1. Added Absolute Path Safety Checks
- Modified `run_raw_task()` to ensure `task_output_dir` is absolute
- Modified `do_inference()` to ensure `task_output_dir` is absolute  
- Modified `dump_cost()` to ensure `task_output_dir` is absolute

### 2. Added Debug Logging
- Added logging in `AgentsManager` to track where Dockerfile, eval.sh, and status.json are written
- Added logging in `TestAnalysisAgent` to track where Dockerfile and eval.sh are written

### 3. Created Debug Script
- Created `debug_file_locations.py` to monitor file creation during execution

## How to Debug the Issue

### Step 1: Run with Debug Logging
The enhanced logging will now show exactly where files are being written. Look for log messages like:
```
Writing Dockerfile to: /path/to/output/dir/Dockerfile
Writing eval.sh to: /path/to/output/dir/eval.sh
Writing status.json to: /path/to/output/dir/status.json
```

### Step 2: Use the Debug Script
Run the debug script in a separate terminal to monitor file creation:

```bash
# In one terminal, start the debug script
python debug_file_locations.py output/swe-factory-runs/kareldb-test 600 10

# In another terminal, run your command
LITELLM_API_BASE="https://api.dev.halo.engineer/v1/ai" \
OPENAI_API_KEY="${OPENAI_API_KEY?->Need a key}" \
PYTHONPATH=. python app/main.py local-issue \
    --task-id "kareldb-connection-1" \
    --local-repo "/Users/hector.maldonado@clearroute.io/xynova/kareldb-cp" \
    --issue-file "input/kareldb_test_issue.txt" \
    --model google/gemini-2.5-flash \
    --output-dir "output/swe-factory-runs/kareldb-test" \
    --setup-dir "output/swe-factory-runs/testbed" \
    --results-path "output/swe-factory-runs/results" \
    --conv-round-limit 3 \
    --num-processes 1 \
    --model-temperature 0.2
```

The debug script will:
- Monitor file creation every 10 seconds for 10 minutes
- Log all new files created
- Warn about files created outside the expected output directory
- Show the current working directory at each check

### Step 3: Check the Logs
Look for:
1. **Expected behavior**: Files being written to the specified output directory
2. **Unexpected behavior**: Files being written to the current working directory or repository root
3. **Working directory changes**: Any unexpected changes in the current working directory

## Expected File Locations

Based on your command, files should be created in:

- **Task output files**: `output/swe-factory-runs/kareldb-test/kareldb-connection-1/`
  - `Dockerfile`
  - `eval.sh`
  - `status.json`
  - `cost.json`
  - `meta.json`
  - `problem_statement.txt`
  - `developer_patch.diff`
  - `info.log`
  - `test_analysis_agent_0/` (subdirectory with test results)

- **Setup directory**: `output/swe-factory-runs/testbed/`
  - Repository clones and working directories

- **Results**: `output/swe-factory-runs/results/results.json`
  - Aggregated results from all tasks

## Troubleshooting

If files are still being created in the wrong location:

1. **Check the debug logs** to see exactly where files are being written
2. **Verify the output directory exists** and is writable
3. **Check for any error messages** about directory creation or file writing
4. **Ensure no other processes** are changing the working directory
5. **Verify the command line arguments** are being parsed correctly

## Additional Recommendations

1. **Use absolute paths** in your command line arguments
2. **Ensure the output directories exist** before running the command
3. **Check file permissions** on the output directories
4. **Monitor system resources** to ensure there are no disk space issues

## Code Changes Summary

The following files were modified to improve file location handling:

- `app/main.py`: Added absolute path safety checks
- `app/agents/agents_manager.py`: Added debug logging for file creation
- `app/agents/test_analysis_agent/test_analysis_agent.py`: Added debug logging for file creation
- `debug_file_locations.py`: Created debug script for monitoring file creation
- `FILE_LOCATION_DEBUG.md`: This documentation file 