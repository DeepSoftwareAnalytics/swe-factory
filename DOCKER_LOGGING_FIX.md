# Docker Logging Configuration Fix

## Issue Description

The Docker containers created by the SWE Factory tool were using a logging configuration that disabled container logs:

```json
"ContainerIDFile": "",
"LogConfig": {
    "Type": "none",
    "Config": {}
}
```

This configuration meant:
- No container logs were being captured
- `docker logs <container>` would not work
- Debugging container issues was difficult
- The tool couldn't capture container output for analysis

## Root Cause

The issue was in the container creation code in two files:
1. `app/agents/test_analysis_agent/docker_utils.py` - `build_container()` function
2. `evaluation/docker_build.py` - `build_container()` and `build_setup_container()` functions

These functions were creating containers without explicit logging configuration, causing Docker to use default settings that might be set to `"none"` in some environments.

## Solution Applied

I've updated all container creation calls to include proper logging configuration:

### Before:
```python
container = client.containers.create(
    image=test_image_name,
    name=test_container_name,
    user="root",
    detach=True,
    command="tail -f /dev/null",
    nano_cpus=None,
    platform="linux/x86_64",
)
```

### After:
```python
container = client.containers.create(
    image=test_image_name,
    name=test_container_name,
    user="root",
    detach=True,
    command="tail -f /dev/null",
    nano_cpus=None,
    platform="linux/x86_64",
    log_config={
        "Type": "json-file",
        "Config": {
            "max-size": "10m",
            "max-file": "3"
        }
    }
)
```

## Benefits of the Fix

1. **Container Logs Available**: You can now use `docker logs <container>` to view container output
2. **Better Debugging**: Container issues can be diagnosed more easily
3. **Log Rotation**: Logs are automatically rotated when they reach 10MB
4. **Storage Management**: Only 3 log files are kept per container
5. **Tool Functionality**: The SWE Factory tool can now capture and analyze container output

## Files Modified

1. `app/agents/test_analysis_agent/docker_utils.py` - Line 330-340
2. `evaluation/docker_build.py` - Lines 590-600 and 650-660

## Testing the Fix

After applying this fix, you should be able to:

1. Run the SWE Factory tool as usual
2. Use `docker logs <container_name>` to view container logs
3. See container output in the tool's log files
4. Debug container issues more effectively

## Recommended Docker Configuration

For optimal performance, ensure your Docker daemon is configured with:

```json
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  }
}
```

This ensures consistent logging behavior across all containers, even those created without explicit logging configuration. 