"""
The main driver.
"""

import json
import shutil
import os
import re
import docker
from argparse import ArgumentParser
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from itertools import chain
from os.path import abspath
from os.path import join as pjoin

from packaging import version
from loguru import logger
from concurrent.futures import TimeoutError
from app import globals, globals_mut, log
from app import utils as apputils
from app.model import common
from app.model.register import register_all_models
from app.agents.agents_manager import AgentsManager
from app.post_process import (
   
    organize_and_form_input,
   
)
from app.raw_tasks import RawGithubTask, RawLocalTask, RawSweTask, RawTask
from app.task import Task
import multiprocessing
import time

def get_args(
    from_command_line_str: str = None, subparser_dest_attr_name: str = "command"
):
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(dest=subparser_dest_attr_name)

    swe_parser = subparsers.add_parser(
        "swe-bench", help="Run one or multiple swe-bench tasks"
    )
    set_swe_parser_args(swe_parser)

    github_parser = subparsers.add_parser(
        "github-issue", help="Run an online github issue"
    )
    set_github_parser_args(github_parser)

    local_parser = subparsers.add_parser("local-issue", help="Run a local issue.")
    set_local_parser_args(local_parser)

    extract_patches_parser = subparsers.add_parser(
        "extract-patches", help="Only extract patches from the raw results dir"
    )
    extract_patches_parser.add_argument("experiment-dir", type=str)

    re_extract_patches_parser = subparsers.add_parser(
        "re-extract-patches",
        help=(
            "same as extract-patches, except that individual dirs"
            " are moved out of their categories first"
        ),
    )
    re_extract_patches_parser.add_argument("experiment-dir", type=str)

    if not from_command_line_str:
        return parser.parse_args()
    return parser.parse_args(from_command_line_str.split())


def main(args, subparser_dest_attr_name: str = "command"):

    ## common options
    globals.output_dir = args.output_dir
    if globals.output_dir is not None:
        globals.output_dir = abspath(globals.output_dir)
    else:
        # Set a default output directory if none is provided
        globals.output_dir = abspath("output/swe-factory-runs/default")
    num_processes: int = int(args.num_processes)
    # set whether brief or verbose log
    print_stdout: bool = not args.no_print
    log.print_stdout = print_stdout
    # model related
    common.set_model(args.model)
    # Set environment variable for subprocesses (keep original model name for OpenAI-compatible endpoints)
    os.environ["SWE_FACTORY_MODEL"] = args.model
    # FIXME: make temperature part of the Model class
    common.MODEL_TEMP = args.model_temperature
    # FIXME: we will remove these hyperparamters, which are from AutoCodeRover, thanks to this work.
    globals.conv_round_limit = args.conv_round_limit
    globals.enable_layered = args.enable_layered
    globals.enable_sbfl = args.enable_sbfl
    globals.enable_validation = args.enable_validation
    globals.enable_angelic = args.enable_angelic
    globals.enable_web_search = args.enable_web_search
    globals.enable_perfect_angelic = args.enable_perfect_angelic
    globals.only_save_sbfl_result = args.save_sbfl_result
    #FIXME  we will remove these hyperparamters, which are from AutoCodeRover, thanks to this work.

    globals.context_generation_limit = args.output_fix_limit
    globals.setup_dir = args.setup_dir 
    if globals.setup_dir is not None:
        globals.setup_dir = abspath(globals.setup_dir)
    else:
        # Set a default setup directory if none is provided
        globals.setup_dir = abspath("output/swe-factory-runs/testbed")
    
    globals.organize_output_only = args.organize_output_only
    # Set a default results_path if none is provided
    if args.results_path is None:
        globals.results_path = None  # Let AgentsManager handle the default
    else:
        globals.results_path = args.results_path
    globals.disable_memory_pool = args.disable_memory_pool
    globals.disable_run_test = args.disable_run_test
    
    globals.disable_context_retrieval= args.disable_context_retrieval
    
    subcommand = getattr(args, subparser_dest_attr_name)
    if subcommand == "swe-bench":
        if globals.organize_output_only:
            organize_and_form_input(globals.output_dir)
            # with docker.DockerClient() as client:
            # client = docker.from_env()
        else:
            client = None
            # try:
            tasks = make_swe_tasks(
                args.task, args.task_list_file,args.tasks_map, args.setup_dir,client
            )
       
            groups = group_swe_tasks_by_env(tasks)
            run_task_groups(groups, num_processes, organize_output=True)
        # finally:
        #     client.close()
    elif subcommand == "github-issue":
        setup_dir = args.setup_dir
        if setup_dir is not None:
            setup_dir = abspath(setup_dir)

        task = RawGithubTask(
            args.task_id,
            args.clone_link,
            args.commit_hash,
            args.issue_link,
            setup_dir,
            args.use_comments,
        )
    
        groups = {"github": [task]}
        run_task_groups(groups, num_processes)
    elif subcommand == "local-issue":
        local_repo = args.local_repo
        if local_repo is not None:
            local_repo = abspath(local_repo)
        issue_file = args.issue_file
        if issue_file is not None:
            issue_file = abspath(issue_file)
        task = RawLocalTask(args.task_id, local_repo, issue_file)
        groups = {"local": [task]}
        run_task_groups(groups, num_processes)
    elif subcommand == "extract-patches":
        organize_and_form_input(globals.output_dir)


def set_swe_parser_args(parser: ArgumentParser) -> None:
    add_task_related_args(parser)

    parser.add_argument(
        "--setup-map",
        type=str,
        help="Path to json file that contains the setup information of the projects.",
    )
    parser.add_argument(
        "--tasks-map",
        type=str,
        help="Path to json file that contains the tasks information.",
    )
    parser.add_argument(
        "--task-list-file",
        default=None,
        type=str,
        help="Path to the file that contains all tasks ids to be run.",
    )
    parser.add_argument(
        "--setup-dir",
        type=str,
        help="The directory where repositories should be cloned to.",
    )
    parser.add_argument(
        "--results-path",
        type=str,
        default=None,
        help="The directory where repositories should be cloned to.",
    )
    parser.add_argument("--task", type=str, help="Task id to be run.")


def set_github_parser_args(parser: ArgumentParser) -> None:
    add_task_related_args(parser)
    parser.add_argument(
        "--task-id", type=str, help="Assign an id to the current fresh issue task."
    )
    parser.add_argument(
        "--clone-link", type=str, help="The link to the repository to clone."
    )
    parser.add_argument(
        "--commit-hash",
        type=str,
        help="The commit hash to checkout. If not specified, the latest commit on default branch will be used.",
    )
    parser.add_argument(
        "--use-comments",
        action="store_true",
        default=False,
        help="Include the comments of the issue.",
    )
   

    parser.add_argument("--issue-link", type=str, help="The link to the issue.")
    parser.add_argument(
        "--setup-dir",
        type=str,
        # default="/home/azureuser/glh/RepoSetupAgent/testbed",
        help="The directory where repositories should be cloned to.",
    )


def set_local_parser_args(parser: ArgumentParser) -> None:
    add_task_related_args(parser)
    parser.add_argument(
        "--task-id", type=str, help="Assign an id to the current local issue task."
    )
    parser.add_argument(
        "--local-repo", type=str, help="Path to a local copy of the target repo."
    )
    parser.add_argument("--issue-file", type=str, help="Path to a local issue file.")
    parser.add_argument(
        "--setup-dir",
        type=str,
        help="The directory where repositories should be cloned to.",
    )
    parser.add_argument(
        "--results-path",
        type=str,
        default=None,
        help="The directory where results should be saved.",
    )


def add_task_related_args(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Path to the directory that stores the run results.",
    )
    parser.add_argument(
        "--no-print",
        action="store_true",
        default=False,
        help="Do not print most messages to stdout.",
    )

    def model_parser(name: str):
        if not isinstance(name, str):
            raise TypeError(f"Invalid model name: {name}")
        if name in common.MODEL_HUB.keys():
            return name
        if name.startswith("litellm-generic-"):
            return name
        # Allow direct model names (like google/gemini-2.5-flash) that contain "/"
        if "/" in name:
            return name
        raise TypeError(f"Invalid model name: {name}")

    parser.add_argument(
        "--model",
        type=model_parser,
        default="gpt-3.5-turbo-0125",
        help="The model to use. Currently only OpenAI models are supported.",
    )
    parser.add_argument(
        "--model-temperature",
        type=float,
        default=0.0,
        help="The model temperature to use, for OpenAI models.",
    )
    parser.add_argument(
        "--conv-round-limit",
        type=int,
        default=15,
        help="Conversation round limit for the main agent.",
    )
    parser.add_argument(
        "--enable-layered",
        action="store_true",
        default=True,
        help="Enable layered code search.",
    )
    parser.add_argument(
        "--enable-sbfl", action="store_true", default=False, help="Enable SBFL."
    )
    parser.add_argument(
        "--enable-web-search", action="store_true", default=False, help="Enable WEB SEARCH."
    )
    parser.add_argument(
        "--enable-validation",
        action="store_true",
        default=False,
        help="Enable validation in our workflow.",
    )
    parser.add_argument(
        "--organize-output-only",
        action="store_true",
        default=False,
        help="Include the comments of the issue.",
    )
    parser.add_argument(
        "--enable-angelic",
        action="store_true",
        default=False,
        help="(Experimental) Enable angelic debugging",
    )
    parser.add_argument(
        "--enable-perfect-angelic",
        action="store_true",
        default=False,
        help="(Experimental) Enable perfect angelic debugging; overrides --enable-angelic",
    )
  
    parser.add_argument(
        "--save-sbfl-result",
        action="store_true",
        default=False,
        help="Special mode to only save SBFL results for future runs.",
    )
    parser.add_argument(
        "--num-processes",
        type=str,
        default=1,
        help="Number of processes to run the tasks in parallel.",
    )
    parser.add_argument(
        "--output-fix-locs",
        action="store_true",
        required=False,
        default=False,
        help="Output fix locations to file and do not repair.",
    )
    parser.add_argument(
        "--output-fix-limit",
        type=int,
        required=False,
        default=10,
        help="Limit output of content retrieval rounds",
    )
  
    parser.add_argument(
        "--disable-memory-pool",
        action="store_true",
        default=False,
        help="Enable layered code search.",
    )
    parser.add_argument(
        "--disable-run-test",
        action="store_true",
        default=False,
        help="Enable layered code search.",
    )
    parser.add_argument(
        "--disable-context-retrieval",
        action="store_true",
        default=False,
        help="Enable layered code search.",
    )


def load_tasks_map(tasks_map_file: str):
    """
    Load a .jsonl or .json file and return a dict: {instance_id: instance_dict}, with original fields only.
    """
    # Detect file type and load raw instances
    if tasks_map_file.endswith('.jsonl'):
        with open(tasks_map_file, 'r', encoding='utf-8') as f:
            instances = [json.loads(line) for line in f if line.strip()]
    else:
        with open(tasks_map_file, 'r', encoding='utf-8') as f:
            obj = json.load(f)
            if isinstance(obj, dict):
                # Already in {id: {...}} format
                return obj
            elif isinstance(obj, list):
                instances = obj
            else:
                raise ValueError("Unsupported JSON structure in file: " + tasks_map_file)
    # Convert to {instance_id: instance_dict}
    return {inst["instance_id"]: inst for inst in instances if "instance_id" in inst}


def make_swe_tasks(
    task_id: str | None,
    task_list_file: str | None,
    # setup_map_file: str,
    tasks_map_file: str,
    setup_dir: str,
    client: docker.DockerClient,
) -> list[RawSweTask]:
    if task_id is not None and task_list_file is not None:
        raise ValueError("Cannot specify both task and task-list.")





    # with open(setup_map_file) as f:
    #     setup_map = json.load(f)
    tasks_map = load_tasks_map(tasks_map_file)
    all_task_ids = []
    if task_list_file is not None:
        all_task_ids = parse_task_list_file(task_list_file)
    if task_id is not None:
        all_task_ids = [task_id]
    if task_list_file is  None:
        all_task_ids = list(tasks_map.keys())
    if len(all_task_ids) == 0:
        raise ValueError("No task ids to run.")
    # with open(tasks_map_file) as f:
    #     tasks_map = json.load(f)

    # Check if all task ids are in the setup and tasks map
    # This allows failing safely if some tasks are not set up properly
    missing_task_ids = [
        x for x in all_task_ids if not (x in tasks_map)
    ]
    if missing_task_ids:
        # Log the tasks that are not in the setup or tasks map
        for task_id in sorted(missing_task_ids):
            log.print_with_time(
                f"Skipping task {task_id} which was not found in setup or tasks map."
            )
        # And drop them from the list of all task ids
        all_task_ids = filter(lambda x: x not in missing_task_ids, all_task_ids)

    all_task_ids = sorted(all_task_ids)

    # for each task in the list to run, create a Task instance
    all_tasks = []
    
 
    # print(len(all_task_ids))
    # input()
    for task_id in all_task_ids:
        setup_info = {}
        task_info = tasks_map[task_id]
        task_start_time_s = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        repo_cache_name = f"{task_info['repo']}_cache"
        repo_cache_dir =  pjoin(setup_dir,repo_cache_name)
        if not os.path.isdir(repo_cache_dir):
            github_link = f"https://github.com/{task_info['repo']}.git"
            apputils.clone_repo_and_checkout(github_link, "", repo_cache_dir)
        else:
            # 可以在这里打印日志或直接跳过
            print(f"Cache already exists: {repo_cache_dir}, skip clone.")
        task_repo_name = f'{task_id}_{task_start_time_s}'
        task_repo_dir =  pjoin(setup_dir,task_repo_name)
        apputils.create_dir_if_not_exists(task_repo_dir)

        setup_info['repo_path'] = task_repo_dir
        setup_info['repo_cache_path'] = repo_cache_dir
        task = RawSweTask(task_id, setup_info, task_info,client)
        all_tasks.append(task)
    # input()
    return all_tasks


def parse_task_list_file(task_list_file: str) -> list[str]:
    """
    Parse the task list file.
    The file should contain one task/instance id per line, without other characters.
    """
    with open(task_list_file) as f:
        task_ids = f.readlines()
    return [x.strip() for x in task_ids]


def group_swe_tasks_by_env(tasks: list[RawSweTask]) -> dict[str, list[RawSweTask]]:
    groups = {}
    for task_index,task in enumerate(tasks):
        # key = task.setup_info["env_name"]
        #we do not run tasks by env.
        key=task_index
        if key not in groups:
            groups[key] = []
        groups[key].append(task)
    return groups


def run_task_groups(
    task_groups: Mapping[str, Sequence[RawTask]],
    num_processes: int,
    organize_output: bool = False,
):
    """
    Main entry for running tasks.
    """
    all_tasks = list(chain.from_iterable(task_groups.values()))
    num_tasks = len(all_tasks)

    globals_mut.init_total_num_tasks(num_tasks)

    # print some info about task
    log.print_with_time(f"Total number of tasks: {num_tasks}")
    log.print_with_time(f"Total number of processes: {num_processes}")
    log.print_with_time(f"Task group info: (number of groups: {len(task_groups)})")
    for key, tasks in task_groups.items():
        log.print_with_time(f"\t{key}: {len(tasks)} tasks")
    
    # single process mode
    if num_processes == 1:
        log.print_with_time("Running in single process mode.")
        run_tasks_serial(all_tasks)
        log.print_with_time("Finished all tasks sequentially.")
    else:
        
        run_task_groups_parallel(task_groups, num_processes)
       

    if globals.only_save_sbfl_result:
        log.print_with_time("Only saving SBFL results. Exiting.")
        return

    if organize_output:
        # post-process completed experiments to get input file to SWE-bench
        log.print_with_time("Post-processing completed experiment results.")
        swe_input_file = organize_and_form_input(globals.output_dir)
        log.print_with_time(f"SWE-Bench input file created: {swe_input_file}")


def run_tasks_serial(tasks: list[RawTask]) -> None:
    for task in tasks:
        run_task_in_subprocess(task)


def run_task_groups_parallel(
    task_groups: Mapping[str, Sequence[RawTask]], num_processes: int
):
    num_task_groups = len(task_groups)
    globals_mut.init_total_num_task_groups(num_task_groups)
    num_processes = min(num_processes, num_task_groups)

    task_group_ids_items = sorted(
        task_groups.items(), key=lambda x: len(x[1]), reverse=True
    )
    log.print_with_time(f"Sorted task groups: {[x[0] for x in task_group_ids_items]}")
    # try:
    #     # Use ProcessPoolExecutor instead of multiprocessing.Pool,
    #     # to support nested sub-processing

    #     group_ids, group_tasks = zip(*task_group_ids_items)
    #     print(group_ids)
    #     print(group_tasks)
    #     with ProcessPoolExecutor(num_processes) as executor:
    #         executor.map(run_task_group, group_ids, group_tasks)
    # except Exception as e:
    #     log.print_with_time(e)
    # finally:
    #     log.print_with_time("Finishing all tasks in the pool.")
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        future_to_gid = {
        executor.submit(_safe_run_group, gid, tasks): gid
        for gid, tasks in task_group_ids_items
        }

        # as_completed yields each Future as soon as it finishes
    for future in as_completed(future_to_gid):
        gid = future_to_gid[future]
        try:
            future.result()
            log.print_with_time(f"Task group {gid} finished successfully.")
        except Exception as e:
            log.print_with_time(f"Task group {gid} failed: {e!r}")

    log.print_with_time("All task groups have been processed.")
    
def _safe_run_group(gid: str, tasks: Sequence[RawTask]) -> None:
    """
    Wrapper to run one task group inside a child process.
    Any exception is re-raised with the group ID for clearer logging.
    """
    try:
        run_task_group(gid, tasks)
    except Exception as e:
        raise RuntimeError(f"Group {gid} execution failed: {e!r}")

def run_task_group(task_group_id: str, task_group_items: list[RawTask]) -> None:
    """
    Run all tasks in a task group sequentially.
    Main entry to parallel processing.
    """
    log.print_with_time(
        f"Starting process for task group {task_group_id}. Number of tasks: {len(task_group_items)}."
    )
    for task in task_group_items:
        # within a group, the runs are always sequential
        run_task_in_subprocess(task)
        log.print_with_time(globals_mut.incre_task_return_msg())

    log.print_with_time(
        f"{globals_mut.incre_task_group_return_msg()} Finished task group {task_group_id}."
    )


# def run_task_in_subprocess(task: RawTask) -> None:
    
#     with ProcessPoolExecutor(max_workers=1) as executor:
#         executor.submit(run_raw_task, task)



# def run_task_in_subprocess(task: RawTask, timeout_seconds: int = 120) -> None:
#     """
#     Run a task in a subprocess, with optional timeout (default: 1 hour).
#     """
#     with ProcessPoolExecutor(max_workers=1) as executor:
#         future = executor.submit(run_raw_task, task)
#         try:
#             future.result(timeout=timeout_seconds)
#         except TimeoutError:
#             log.log_and_always_print(f"[TIMEOUT] Task {task.task_id} exceeded {timeout_seconds} seconds and was terminated.")
#         except Exception as e:
#             log.log_and_always_print(f"[ERROR] Task {task.task_id} failed with exception: {e}")



def run_task_in_subprocess(task: RawTask, timeout_seconds: int = 5400) -> None:
    """
    Run a task in a subprocess, with hard timeout control.
    """
    p = multiprocessing.Process(target=run_raw_task, args=(task,))
    p.start()
    p.join(timeout=timeout_seconds)
    if p.is_alive():
        log.log_and_always_print(f"[TIMEOUT] Task {task.task_id} exceeded {timeout_seconds}s. Killing it...")
        p.terminate()
        p.join()

def run_raw_task(
    task: RawTask, print_callback: Callable[[dict], None] | None = None
) -> bool:
    """
    High-level entry for running one task.

    Args:
        - task: The Task instance to run.

    Returns:
        Whether the task completed successfully.
    """
    # Set the model in the subprocess to ensure SELECTED_MODEL is available
    # We need to get the model name from the current process environment or use a default
    model_name = os.getenv("SWE_FACTORY_MODEL", "gpt-3.5-turbo-0125")
    common.set_model(model_name)
    
    task_id = task.task_id

    start_time_s = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # task_output_dir = pjoin(globals.output_dir, f"{task_id}_{start_time_s}")
    
    # Ensure globals.output_dir is set
    if not globals.output_dir:
        globals.output_dir = abspath("output/swe-factory-runs/default")
    
    task_output_dir = pjoin(globals.output_dir, f"{task_id}")
    
    # Ensure task_output_dir is absolute
    task_output_dir = os.path.abspath(task_output_dir)
    
    status_file = pjoin(task_output_dir, "status.json")
    if os.path.exists(status_file):
        log.log_and_always_print(f"Status file already exists for task {task_id}, skipping execution")
        return True
    elif os.path.exists(task_output_dir):
        # If directory exists but no status.json, clean it up
        try:
            shutil.rmtree(task_output_dir)
            log.log_and_always_print(f"Cleared existing task directory {task_output_dir} as it had no status.json")
        except Exception as e:
            log.log_and_always_print(f"Error clearing task directory {task_output_dir}: {e}")
            return False
    apputils.create_dir_if_not_exists(task_output_dir)
    task.dump_meta_data(task_output_dir)

    log.log_and_always_print(f"============= Running task {task_id} =============")

    run_ok = False

    try:
        run_ok = do_inference(task.to_task(), task_output_dir, print_callback)

        if run_ok:
            run_status_message = f"Task {task_id} completed successfully."
        else:
            run_status_message = f"Task {task_id} failed without exception."
    except Exception as e:
        logger.exception(e)
        run_status_message = f"Task {task_id} failed with exception: {e}."

    log.log_and_always_print(run_status_message)

    return run_ok


def do_inference(
    python_task: Task,
    task_output_dir: str,
    print_callback: Callable[[dict], None] | None = None,
) -> bool:
    client = docker.from_env()
    
    # Ensure task_output_dir is absolute
    task_output_dir = os.path.abspath(task_output_dir)
    
    apputils.create_dir_if_not_exists(task_output_dir)
    # github_link = f'https://github.com/{python_task.repo_name}.git'
    # Handle both SweTask and PlainTask
    if hasattr(python_task, 'commit'):
        commit_hash = python_task.commit
    elif hasattr(python_task, 'commit_hash'):
        commit_hash = python_task.commit_hash
    else:
        raise AttributeError(f"Task object {type(python_task)} has no commit or commit_hash attribute")
    
    # For PlainTask, create a working directory in the testbed
    if hasattr(python_task, 'repo_cache_path'):
        repo_cache_path = python_task.repo_cache_path
    else:
        # Ensure globals.setup_dir is set
        if not globals.setup_dir:
            globals.setup_dir = abspath("output/swe-factory-runs/testbed")
        
        # Create a working directory in the testbed to avoid deleting the original repo
        working_dir = pjoin(globals.setup_dir, f"{python_task.project_path.split('/')[-1]}_working")
        repo_cache_path = python_task.project_path
        python_task.project_path = working_dir
    
    apputils.clone_repo_and_checkout(repo_cache_path, commit_hash, python_task.project_path)
    logger.add(
        pjoin(task_output_dir, "info.log"),
        level="DEBUG",
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level>"
            " | <level>{message}</level>"
        ),
    )

    start_time = datetime.now()

    
    try:
        agents_manager = AgentsManager(python_task, 
                                        task_output_dir,
                                        client,
                                        start_time,
                                        globals.conv_round_limit,
                                        globals.results_path,
                                        disable_memory_pool = globals.disable_memory_pool,
                                        disable_context_retrieval= globals.disable_context_retrieval,
                                        disable_run_test= globals.disable_run_test
                                        )
        agents_manager.run_workflow()
        run_ok = True
        end_time = datetime.now()

        dump_cost(start_time, end_time, task_output_dir, python_task.project_path)
    finally:
        # python_task.reset_project()
        # python_task.remove_project()  # Commented out to prevent deleting original repository
        if client:
            client.close()

    return run_ok


def dump_cost(
    start_time: datetime, end_time: datetime, task_output_dir: str, project_path: str
):
    # Ensure task_output_dir is absolute
    task_output_dir = os.path.abspath(task_output_dir)
    
    with apputils.cd(project_path):
        commit_hash = apputils.get_current_commit_hash()
    model_stats = common.SELECTED_MODEL.get_overall_exec_stats()
    stats = {
        "commit": commit_hash,
        "start_epoch": start_time.timestamp(),
        "end_epoch": end_time.timestamp(),
        "elapsed_seconds": (end_time - start_time).total_seconds(),
    }
    stats.update(model_stats)

    with open(pjoin(task_output_dir, "cost.json"), "w") as f:
        json.dump(stats, f, indent=4)


if __name__ == "__main__":
    logger.remove()
    register_all_models()
    args = get_args()
    main(args)
