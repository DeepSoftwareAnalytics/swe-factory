# editagent_script.py

import openai
import re
import yaml
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime
import json
import concurrent.futures
import threading
import docker

from inference.agenthub.runtime.docker import DockerRuntime
from inference.agenthub.environment.env import EnvArgs, RepoEnv
from inference.agenthub.agent.agent import AgentArgs, Agent

from inference.docker_bash_utils.docker_list_tags import fetch_docker_tags
from inference.agenthub.utils.log import get_logger
from inference.logging import setup_logging, INFO
from inference.agenthub.utils.utils import get_parsed_commit

from fire import Fire
from inference.agenthub.utils.utils import match_dockerimage_to_repo
from inference.agenthub import SUPPORTED_REPOS
from datasets import load_dataset
from inference.agenthub.trajectory import TrajectoryStep, Trajectory
import time

##############################################################################
# Initialize Logger
##############################################################################
logger = get_logger(__name__)  # Initialize the logger

##############################################################################
# Initialize File Lock for Thread-Safe Writing
##############################################################################
file_lock = threading.Lock()


##############################################################################
# Utility Function
##############################################################################
TIMEOUT_SIGNATURE = "The command took too long to execute"

def resolve_instance_metadata(ds: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize dataset fields that we need when storing per-instance artifacts.
    """
    docker_image = ds.get("docker_image", "unknown_docker_image")
    instance_id = ds.get("instance_id")
    if not instance_id:
        instance_id = docker_image.split("_")[0] if "_" in docker_image else docker_image

    repo = ds.get("repo") or ds.get("repository")
    issue_number = ds.get("issue_number")
    if issue_number is None:
        issue_numbers = ds.get("issue_numbers")
        if isinstance(issue_numbers, list) and issue_numbers:
            issue_number = issue_numbers[0]

    return {
        "instance_id": instance_id,
        "docker_image": docker_image,
        "repo": repo,
        "issue_number": issue_number,
    }


def write_instance_artifacts(
    instance_dir: Path,
    trajectory: Trajectory,
    instance_meta: Dict[str, Any],
    run_logger,
) -> None:
    """
    Persist metadata + reward/test outputs for downstream inspection.
    """

    def _write_text_file(path: Path, value: Optional[str]) -> None:
        path.write_text(value or "", encoding="utf-8")

    test_timeout = bool(
        trajectory.test_output and TIMEOUT_SIGNATURE in trajectory.test_output
    )

    last_step = trajectory.trajectory_steps[-1] if trajectory.trajectory_steps else None
    conversation_total = last_step.token_usage_total if last_step else trajectory.num_tokens_total

    token_usage_summary = {
        "input": trajectory.num_tokens_prompt,
        "output": trajectory.num_tokens_completion,
        "total": trajectory.num_tokens_total,
        "conversation_total": conversation_total,
    }

    metadata_payload = {
        "instance_id": instance_meta["instance_id"],
        "docker_image": instance_meta["docker_image"],
        "repo": instance_meta.get("repo"),
        "issue_number": instance_meta.get("issue_number"),
        "reward": trajectory.reward,
        "reward_calc_time": trajectory.reward_calc_time,
        "exit_reason": trajectory.exit_reason,
        "exp_name": trajectory.exp_name,
        "test_timeout": test_timeout,
        "token_usage": token_usage_summary,
    }

    try:
        (instance_dir / "metadata.json").write_text(
            json.dumps(metadata_payload, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:
        run_logger.error(f"Failed to write metadata.json for {instance_dir}: {exc}")

    try:
        _write_text_file(instance_dir / "output_patch.diff", trajectory.output_patch)
    except Exception as exc:
        run_logger.error(f"Failed to write output_patch.diff for {instance_dir}: {exc}")

    try:
        _write_text_file(instance_dir / "test_output.log", trajectory.test_output)
    except Exception as exc:
        run_logger.error(f"Failed to write test_output.log for {instance_dir}: {exc}")


def get_docker_images(repo_name) -> List[str]:
    """
    Fetches the list of Docker images available for the base image.

    Returns:
        A list of Docker image tags.
    """
    base_image = f"namanjain12/{repo_name}new"
    tags = fetch_docker_tags(base_image)
    docker_image_list = [f"{base_image}:{x['name']}" for x in tags]
    return docker_image_list


def prepull_docker_image(docker_image: str) -> bool:
    """
    Prepulls a single Docker image.
    
    Args:
        docker_image: The Docker image name to pull
        
    Returns:
        True if successful, False otherwise
    """
    try:
        client = docker.from_env()
        logger.info(f"Pulling Docker image: {docker_image}")
        client.images.pull(docker_image)
        logger.info(f"Successfully pulled Docker image: {docker_image}")
        return True
    except Exception as e:
        logger.error(f"Failed to pull Docker image {docker_image}: {e}")
        return False


def prepull_docker_images(ds_selected: List[Dict], max_workers: Optional[int] = None) -> None:
    """
    Prepulls all Docker images in parallel before starting the main execution.
    
    Args:
        ds_selected: List of dataset entries containing docker_image keys
        max_workers: Maximum number of threads for parallel pulling
    """
    # Extract unique Docker images
    docker_images = list(set([ds_entry["docker_image"] for ds_entry in ds_selected]))
    logger.info(f"Starting parallel prepull of {len(docker_images)} unique Docker images...")
    
    # Use ThreadPoolExecutor for I/O bound operations like Docker pulls
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all pull tasks
        future_to_image = {
            executor.submit(prepull_docker_image, docker_image): docker_image
            for docker_image in docker_images
        }
        
        # Track results
        successful_pulls = []
        failed_pulls = []
        
        for future in concurrent.futures.as_completed(future_to_image):
            docker_image = future_to_image[future]
            try:
                success = future.result()
                if success:
                    successful_pulls.append(docker_image)
                else:
                    failed_pulls.append(docker_image)
            except Exception as e:
                logger.error(f"Exception during prepull of {docker_image}: {e}")
                failed_pulls.append(docker_image)
    
    logger.info(f"Prepull completed. Success: {len(successful_pulls)}, Failed: {len(failed_pulls)}")
    if failed_pulls:
        logger.warning(f"Failed to pull images: {failed_pulls}")


def postprocess_trajectories_history_only(exp_name: str, traj_dir_path: Path, run_logger) -> None:
    """
    Emit auxiliary trajectory files containing only the `history` payload per line:
    - trajectories.jsonl: all histories
    - trajectories_rejection_sampling.jsonl: reward == 1 histories
    - reward_summary.json: counts and instance_id lists for reward 1/0

    If a tools schema is present (fn-calling), wrap each line as:
    {"messages": [...], "tools": [...]}
    """
    src = traj_dir_path / f"{exp_name}.jsonl"
    if not src.exists():
        run_logger.warning(f"skip postprocess: missing {src}")
        return

    lines = [ln.strip() for ln in src.read_text().splitlines() if ln.strip()]
    if not lines:
        run_logger.warning(f"skip postprocess: empty {src}")
        return

    records = []
    for ln in lines:
        try:
            obj = json.loads(ln)
            history = obj.get("history")
            if history is None:
                run_logger.warning("postprocess: missing history; skip line")
                continue
            agent_args = obj.get("agent_args") or {}
            other_args = agent_args.get("other_args") or {}
            tools = other_args.get("tools_schema")
            records.append(
                {
                    "history": history,
                    "tools": tools,
                    "reward": obj.get("reward"),
                    "inst_id": (obj.get("ds") or {}).get("instance_id") or (obj.get("ds") or {}).get("docker_image"),
                }
            )
        except Exception as exc:
            run_logger.error(f"postprocess parse error: {exc}")

    if not records:
        run_logger.warning("postprocess: no valid records")
        return

    traj_path = traj_dir_path / "trajectories.jsonl"
    traj_r1_path = traj_dir_path / "trajectories_rejection_sampling.jsonl"
    summary_path = traj_dir_path / "reward_summary.json"

    reward1_ids, reward0_ids = [], []

    with traj_path.open("w", encoding="utf-8") as f_all, traj_r1_path.open("w", encoding="utf-8") as f_r1:
        for rec in records:
            if rec["tools"] is not None:
                payload = {"messages": rec["history"], "tools": rec["tools"]}
            else:
                payload = rec["history"]
            h_line = json.dumps(payload)
            f_all.write(h_line + "\n")
            if rec["reward"] == 1:
                f_r1.write(h_line + "\n")
                if rec["inst_id"]:
                    reward1_ids.append(rec["inst_id"])
            else:
                if rec["inst_id"]:
                    reward0_ids.append(rec["inst_id"])

    summary = {
        "total": len(records),
        "reward_1": {"count": len(reward1_ids), "instance_ids": reward1_ids},
        "reward_0": {"count": len(reward0_ids), "instance_ids": reward0_ids},
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


##############################################################################
# editagent Functions
##############################################################################
def run_agent_with_restarts(
    agent,
    env,
    max_steps=40,
    num_restarts=1,
    temperature=0.0,
    max_steps_absolute=50,
    use_fn_calling: bool = True,
    max_iterations: int = 1,
    scaffold: str = "r2egym",
    max_tokens: int = 65536,
):
    """
    Iterative eval protocol:
    - normally run the agent
    - run for maximum num_iterations = 3 times
    - stop if trajectory.exit_reason == "agent"
    - otherwise continue iteratively till maximum iterations
    - finally choose the trajectory with the lowest number of steps
    - note restarts and iterative_evals are different (so just use one of them | add an assert flag)
    - also if original is at temp = 0, then we do next with 0.1 and 0.2 and so on (max 0.2)
    """
    steps_per_agent = max_steps // num_restarts
    logger.warning(f"running {steps_per_agent} steps per agent")

    # only one of restarts > 1 and iterative_eval can be True
    iterative_eval = max_iterations > 1
    assert not (num_restarts > 1 and iterative_eval), "only one of restarts > 1 and iterative_eval can be True"
    logger.warning(f"Using iterations: {max_iterations}, using iterative protocol: {iterative_eval}")

    # if original is at temp = 0, then we do next with 0.1 and 0.2 and so on (max 0.2)
    # if temperature is 0, create list of increasing temperatures up to 0.2
    if temperature == 0:
        temperatures = [0.0 + 0.1 * i for i in range(max_iterations)]
        temperatures = [min(t, 0.2) for t in temperatures]  # cap at 0.2
    else:
        temperatures = [temperature] * max_iterations
    logger.warning(f"Using temperatures: {temperatures}")
    history = None
    # run the agent in iterative protocol
    trajectories = []
    for iteration in range(max_iterations):
        for idx in range(num_restarts):
            logger.warning(f"running agent at idx: {idx+1}")
            trajectory = agent.run(
                env,
                max_steps=steps_per_agent,
                temperature=temperatures[iteration],
                max_steps_absolute=max_steps_absolute,
                use_fn_calling=use_fn_calling,
                scaffold=scaffold,
                max_token_limit=max_tokens,
            )
            history = agent.history
            # remove reproduce.py
            # env.runtime.run('rm reproduce_issue.py')
        if trajectory.exit_reason == "agent":
            logger.warning(f"agent self-finished at iteration: {iteration}")
            return trajectory,history
        # otherwise continue iteratively
        trajectories.append(trajectory)
        # reset the env
        # env.reset()

    # choose the trajectory with the lowest number of steps
    trajectory = min(trajectories, key=lambda x: x.num_steps)
    return trajectory, history


def runagent(
    ds,
    exp_name: Optional[str] = None,
    max_steps=40,
    num_restarts=1,
    max_steps_absolute=50,
    llm_name="gpt-4o",
    temperature=0,
    use_fn_calling: bool = True,
    backend: str = "kubernetes", # "kubernetes" or "docker"
    max_reward_calc_time: int = 300,
    max_iterations: int = 1,
    scaffold: str = "r2egym",
    max_tokens: int = 65536,
    root_mode: bool = True,
) -> Optional[str]:
    """
    Runs the editagent agent on a specified Docker image.

    Args:
        docker_image: The Docker image to use for the environment.
        traj_dir: Directory to save trajectories.
        jsonl_file: Path to the JSONL file to save results. If not provided, generated using traj_dir and exp_name.
        exp_name: Experiment name. Used if jsonl_file is not provided. If not provided, a unique name is generated.
    """
    assert scaffold in ["r2egym", "sweagent", "openhands", "mini_swe_agent", "live_swe_agent"], (
        f"Scaffold is {scaffold}, must be one of [r2egym, sweagent, openhands, mini_swe_agent, live_swe_agent]"
    )
    # Generate a unique experiment name if not provided
    if exp_name is None:
        exp_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    instance_meta = resolve_instance_metadata(ds)
    instance_dir = Path("run_logs") / exp_name / instance_meta["instance_id"]
    instance_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(
        name=ds["docker_image"].replace("/", "_"),
        log_file=str(instance_dir / "agent.log"),
        console=True,
        level=INFO,
    )
    logger.info(f"Starting editagent on Docker image: {ds['docker_image']}")
    logger.info(f"Using LLM: {llm_name}")
    logger.info(f"Max Steps: {max_steps}")

    # Initialize environment arguments
    env_args = EnvArgs(ds=ds, root_mode=root_mode)

    # Initialize the RepoEnv
    if scaffold in ["mini_swe_agent", "live_swe_agent"]:
        env = RepoEnv(env_args, logger=logger, backend=backend, scaffold=scaffold, step_timeout=60)
    else:
        env = RepoEnv(env_args, logger=logger, backend=backend, scaffold=scaffold)
    # set agent args
    if use_fn_calling:
        assert scaffold != "sweagent", "SWEagent scaffold does not support fn calling"
        assert scaffold not in ["mini_swe_agent", "live_swe_agent"], "mini_swe_agent/live_swe_agent scaffolds are non-fn-calling only"
        agent_args = AgentArgs.from_yaml(
            Path(f"./inference/agenthub/config/{scaffold}/edit_fn_calling.yaml")
        )
    else:
        agent_args = AgentArgs.from_yaml(
            Path(f"./inference/agenthub/config/{scaffold}/edit_non_fn_calling.yaml")
        )
    agent_args.llm_name = llm_name

    # Initialize the agent
    agent = Agent(name="EditAgent", args=agent_args, logger=logger)

    try:
        # run agent editagent
        try:
            trajectory,history = run_agent_with_restarts(
                agent,
                env,
                max_steps=max_steps,
                num_restarts=num_restarts,
                temperature=temperature,
                max_steps_absolute=max_steps_absolute,
                use_fn_calling=use_fn_calling,
                max_iterations=max_iterations,
                scaffold=scaffold,
                max_tokens=max_tokens,
            )
        except Exception as e:
            logger.error(
                f"Error during agent run for Docker image {ds['docker_image']}: {e}"
            )
            return None

        # also get the gt outputs
        reward_calc_time = time.time()
        try:
            reward, test_output = env.runtime._calculate_reward(get_test_output=True, timeout=max_reward_calc_time)
        except Exception as e:
            logger.error(
                f"Error calculating reward for Docker image {ds['docker_image']}: {e}"
            )
            return None
        reward_calc_time = time.time() - reward_calc_time

        # update the trajectory object
        trajectory.reward = reward
        trajectory.test_output = test_output
        trajectory.ds = ds
        trajectory.exp_name = exp_name
        trajectory.reward_calc_time = reward_calc_time # time taken to calculate reward
        trajectory.history = history
        
        logger.warning(f"time taken to calculate reward in seconds: {reward_calc_time:.2f}")

        write_instance_artifacts(instance_dir, trajectory, instance_meta, logger)

        logger.info(f"editagent completed for Docker image: {ds['docker_image']}")
        return trajectory.model_dump_json()
    finally:
        # close env and docker runtime
        logger.info(f"Closing environment for Docker image: {ds['docker_image']}")
        try:
            env.close()
        except Exception as e:
            logger.error(
                f"Error closing environment for Docker image {ds['docker_image']}: {e}"
            )

def runagent_multiple(
    dataset: str,
    split: str,
    k: int = 1,
    traj_dir: str = "./traj",
    exp_name: Optional[str] = None,
    start_idx=0,
    max_steps=40,
    num_restarts=1,
    max_steps_absolute=50,
    max_workers: Optional[int] = None,
    llm_name="gpt-4o",
    use_existing: bool = True,
    skip_existing: bool = False,
    temperature: float = 0,
    use_fn_calling: bool = True,
    backend: str = "kubernetes", # "kubernetes" or "docker"
    max_reward_calc_time: int = 300,
    max_iterations: int = 1,
    scaffold: str = "r2egym",
    prepull_images: bool = False,
    max_tokens: int = 65536,
    root_mode: bool = True,
):
    """
    Runs the editagent agent on the first k Docker images.

    Args:
        k: The number of Docker images to process.
        traj_dir: Directory to save trajectories.
        exp_name: Experiment name for the JSONL file. If not provided, a unique name is generated.
        start_idx: The starting index in the Docker images list.
        max_steps: Maximum steps for the agent run.
        max_workers: Maximum number of threads to use.
        prepull_images: Whether to prepull Docker images in parallel before starting execution.
    """
    # Allow mini_swe_agent/live_swe_agent as scaffolds
    assert scaffold in ["r2egym", "sweagent", "openhands", "mini_swe_agent", "live_swe_agent"], (
        f"Scaffold is {scaffold}, must be one of [r2egym, sweagent, openhands, mini_swe_agent, live_swe_agent]"
    )

    # Load the dataset
    if dataset.endswith('.json'):
        ds = load_dataset("json", data_files={split: dataset})[split]
    else:
        ds = load_dataset(dataset, split=split)
    logger.info(f"{len(ds)}, {k}, {start_idx}")
    # shuffle the dataset
    # ds = ds.shuffle(seed=42)

    # get selected idxs
    selected_idx = range(start_idx, start_idx + k)
    ds_selected = [ds[i] for i in selected_idx]

    # print ds_selected stats
    logger.info(
        f"Dataset: {dataset}, Split: {split}, Num_total: {len(ds)}, Start Index: {start_idx}, k: {k}"
    )
    logger.info(f"Starting editagent on {len(ds_selected)} Docker images.")

    # Generate a unique experiment name if not provided
    if exp_name is None:
        exp_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Ensure traj_dir exists
    traj_dir_path = Path(traj_dir)
    traj_dir_path.mkdir(parents=True, exist_ok=True)

    # Generate a filename for the JSONL file
    jsonl_file = traj_dir_path / f"{exp_name}.jsonl"

    if use_existing:
        if jsonl_file.exists():
            with open(jsonl_file) as f:
                existing_dockers = []
                for line in f.readlines():
                    try:
                        existing_dockers.append(
                            Trajectory.load_from_model_dump_json(line).ds[
                                "docker_image"
                            ]
                        )
                    except:
                        print("error in jsonl file")

            ds_selected = [
                ds_entry
                for ds_entry in ds_selected
                if ds_entry["docker_image"] not in existing_dockers
            ]

    if skip_existing:
        old_jsonl_files_glob = f"{exp_name[:-1]}*"
        for old_jsonl_file in traj_dir_path.glob(old_jsonl_files_glob):
            with open(old_jsonl_file) as f:
                existing_dockers = [
                    loadline["ds"]["docker_image"]
                    for line in f
                    for loadline in [json.loads(line)]
                    if loadline["reward"] == 1
                ]

            ds_selected = [
                ds_entry
                for ds_entry in ds_selected
                if ds_entry["docker_image"] not in existing_dockers
            ]

    logger.info(
        f"Starting editagent on {len(ds_selected)} Docker images after filtering."
    )

    # Prepull all Docker images in parallel before starting main execution
    if ds_selected and prepull_images:
        logger.info("Prepulling Docker images before starting main execution...")
        prepull_docker_images(ds_selected, max_workers=max_workers)
        logger.info("Docker image prepull completed.")

    # with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks to the executor using keyword arguments
        future_to_image = {
            executor.submit(
                runagent,
                ds=ds_entry,
                exp_name=exp_name,
                max_steps=max_steps,
                num_restarts=num_restarts,
                max_steps_absolute=max_steps_absolute,
                llm_name=llm_name,
                temperature=temperature,
                use_fn_calling=use_fn_calling,
                backend=backend,
                max_reward_calc_time=max_reward_calc_time,
                max_iterations=max_iterations,
                scaffold=scaffold,
                max_tokens=max_tokens,
                root_mode=root_mode,
            ): ds_entry[
                "docker_image"
            ]  # <-- store the docker_image from ds_entry here
            for ds_entry in ds_selected
        }

        with open(jsonl_file, "a") as f:
            for future in concurrent.futures.as_completed(future_to_image):
                docker_image = future_to_image[
                    future
                ]  # <-- retrieve that stored docker_image
                try:
                    result = future.result()
                    if result is not None:
                        with file_lock:
                            f.write(result + "\n")
                except Exception as e:
                    # Use docker_image from above when logging
                    logger.error(f"Exception for Docker image {docker_image}: {e}")

    # Produce auxiliary history-only files for finetuning/rejection sampling.
    postprocess_trajectories_history_only(exp_name, traj_dir_path, logger)
    logger.info(f"editagent completed on {len(ds_selected)} Docker images.")


if __name__ == "__main__":
    # Expose functions via Fire
    Fire(
        {
            "runagent": runagent,
            "runagent_multiple": runagent_multiple,
        }
    )
