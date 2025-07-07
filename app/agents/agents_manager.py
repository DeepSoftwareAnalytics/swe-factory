from app.data_structures import MessageThread
from app.task import Task
from app.agents.write_dockerfile_agent import WriteDockerfileAgent
from app.agents.test_analysis_agent import TestAnalysisAgent
from app.agents.write_eval_script_agent import WriteEvalScriptAgent
from app.agents.context_retrieval_agent import ContextRetrievalAgent
import os
import re
import docker
from datetime import datetime
from app.model import common
from os.path import join as pjoin
from loguru import logger
from packaging import version
import json
import random
from filelock import FileLock
from copy import deepcopy

DIFF_MODIFIED_FILE_REGEX = r"--- a/(.*)"
def normalize_version(ver_str):
    match = re.search(r"(\d+(?:\.\d+){0,2})", ver_str)
    return match.group(1) if match else ver_str

def get_closest_version_info(records, repo, target_version):
    same_repo = [r for r in records if r.get('repo') == repo]
    if not same_repo:
        return None
    ver_map = {r['version']: normalize_version(r['version']) for r in same_repo}
    try:
        sorted_list = sorted(same_repo, key=lambda r: version.parse(ver_map[r['version']]))
        target_parsed = version.parse(normalize_version(target_version))
    except Exception:
        sorted_list = sorted(same_repo, key=lambda r: r['version'])
        target_parsed = version.parse(normalize_version(target_version))
    exact_matches = [r for r in same_repo if r['version'] == target_version]
    if exact_matches:
        return random.choice(exact_matches)
    candidates = [r for r in sorted_list if version.parse(ver_map[r['version']]) <= target_parsed]
    return random.choice(candidates) if candidates else None

class AgentsManager:
    """
    Simple manager to orchestrate LLM-based agents.
    """
    def __init__(self, 
                task: Task, 
                output_dir: str,
                client: docker.DockerClient, 
                start_time: datetime, 
                max_iteration_num: int,
                results_path: str | None,
                disable_memory_pool:bool,
                disable_context_retrieval:bool,
                disable_run_test:bool,
                ):
        self.task = task
        self.output_dir = os.path.abspath(output_dir)
        self.run_count = 0
        self.client = client
        self.max_iteration_num  = max_iteration_num
        self.start_time = start_time
        
        self.test_files = self.get_test_files()
        self.repo_basic_info = self.get_repository_basic_info()
        self.workflow_finish_status  = False
        # Initialize agents
        self.agents_dict = {
            "write_docker_agent": WriteDockerfileAgent(task, output_dir, self.repo_basic_info),
            "write_eval_script_agent": WriteEvalScriptAgent(task, output_dir, self.repo_basic_info),
            "test_analysis_agent": TestAnalysisAgent(task, output_dir, self.repo_basic_info, client),
            "context_retrieval_agent": ContextRetrievalAgent(task, output_dir, self.repo_basic_info),
        }
        self.set_agent_status('all',False)
        self.disable_memory_pool = disable_memory_pool
        self.disable_context_retrieval = disable_context_retrieval
        self.disable_run_test = disable_run_test
        if disable_context_retrieval:
            self.set_agent_status("context_retrieval_agent",True)
        self.agents_dict['test_analysis_agent'].disable_context_retrieval= disable_context_retrieval
        self.agents_dict['test_analysis_agent'].disable_run_test = disable_run_test
        
        # Handle None results_path by setting a default
        if results_path is None:
            results_path = os.path.join(output_dir, "results")
        
        # Ensure results_path is absolute
        if not os.path.isabs(results_path):
            results_path = os.path.abspath(results_path)
        
        # Create the results directory if it doesn't exist
        os.makedirs(results_path, exist_ok=True)
        
        self.results_file = os.path.join(results_path, 'results.json')
        lock_path = self.results_file + '.lock'
        self.lock = FileLock(lock_path, timeout=30)
        with self.lock:
            if not os.path.exists(self.results_file):
                with open(self.results_file, 'w') as f:
                    json.dump([], f, indent=2)

    def set_agent_status(self, agent_name: str, status: bool):
        """Set the status of an agent to control if it's active or inactive."""
        if agent_name == 'all':
            for agent_key, agent_value in self.agents_dict.items():
                agent_value.finish_status = status  

        elif agent_name in self.agents_dict:
            agent = self.agents_dict[agent_name]
            agent.finish_status = status  
        else:
            logger.error(f"Agent {agent_name} not found!")

    def get_agent_status(self, agent_name: str) -> bool:
        """Get the current status of an agent."""
        if agent_name in self.agents_dict:
            return self.agents_dict[agent_name].finish_status
        else:
            logger.error(f"Agent {agent_name} not found!")
            return False

    def set_agents_iteration_num(self, iteration_num: int) -> None:
        """Get the current status of an agent."""
        for agent_key, agent_value in self.agents_dict.items():
            
            agent_value.iteration_num = iteration_num 
            
    def get_test_files(self) -> list[str]:
        test_files = re.findall(DIFF_MODIFIED_FILE_REGEX, self.task.test_patch)
        return test_files
    
    def get_repository_basic_info(self) -> str:
        return (
            f"Target repository name: {self.task.repo_name}\n"
            f"Commit SHA: {self.task.commit}\n"
            f"Version: {self.task.version}\n"
            "Target test files:\n"
            + "\n".join(self.test_files)
            + "\n"
        )

    def dump_cost(
        self
    ):
        start_time = self.start_time
        end_time = datetime.now()
        task_output_dir = self.output_dir
        project_path  = self.task.project_path
      
        model_stats = common.SELECTED_MODEL.get_overall_exec_stats()
        stats = {
            # "commit": commit_hash,
            "start_epoch": start_time.timestamp(),
            "end_epoch": end_time.timestamp(),
            "elapsed_seconds": (end_time - start_time).total_seconds(),
        }
        stats.update(model_stats)

        with open(pjoin(task_output_dir, "cost.json"), "w") as f:
            json.dump(stats, f, indent=4)

    def _read_results(self) -> list:
        with self.lock:
            with open(self.results_file, "r") as f:
                return json.load(f)

    def _write_results(self, records: list) -> None:
        tmp = self.results_file + ".tmp"
        with self.lock:
            with open(tmp, "w") as f:
                json.dump(records, f, indent=2)
            os.replace(tmp, self.results_file)

    def get_latest_reference_setup_for_repo(self):
        records = self._read_results()
        return get_closest_version_info(records, self.task.repo_name, self.task.version)

    def run_workflow(self) -> None:
        for iteration_num in range(self.max_iteration_num):
            self.set_agents_iteration_num(iteration_num)
            
            if self.disable_context_retrieval and iteration_num==0:
              readme_content = self.agents_dict['context_retrieval_agent'].browse_readme()
              if readme_content:
                  self.agents_dict['write_eval_script_agent'].add_user_message(readme_content)
                  self.agents_dict['write_docker_agent'].add_user_message(readme_content)
            

            if not self.get_agent_status("context_retrieval_agent"):
                collected_information, summary, success =  self.agents_dict['context_retrieval_agent'].run_task()
                self.dump_cost()
                if collected_information != None:
                    self.set_agent_status("context_retrieval_agent",True)
                    self.agents_dict['write_eval_script_agent'].add_user_message(collected_information)
                    self.agents_dict['write_docker_agent'].add_user_message(collected_information)
                    
            if self.disable_memory_pool == False:        
                reference_setup = self.get_latest_reference_setup_for_repo()
                if reference_setup:
                    self.agents_dict['write_docker_agent'].reference_setup = reference_setup
                    
                    self.agents_dict['write_eval_script_agent'].reference_setup = reference_setup

            if self.get_agent_status("context_retrieval_agent") and not self.get_agent_status("write_docker_agent"):
                _, _, success =  self.agents_dict['write_docker_agent'].run_task()
                self.dump_cost()
                if success:
                    self.set_agent_status("write_docker_agent",True)
          

            if self.get_agent_status("context_retrieval_agent") and self.get_agent_status("write_docker_agent") and not self.get_agent_status("write_eval_script_agent"):
                self.agents_dict['write_eval_script_agent'].dockerfile =  self.agents_dict['write_docker_agent'].get_latest_dockerfile()
                _, _, success =  self.agents_dict['write_eval_script_agent'].run_task()
                self.dump_cost()
                if success:
                    self.set_agent_status("write_eval_script_agent",True)
                
            if self.get_agent_status("context_retrieval_agent") and self.get_agent_status("write_docker_agent") and self.get_agent_status("write_eval_script_agent"):
                dockerfile = self.agents_dict['write_docker_agent'].get_latest_dockerfile()
                eval_script_skeleton = self.agents_dict['write_eval_script_agent'].get_latest_eval_script_skeleton()
                eval_script= self.agents_dict['write_eval_script_agent'].get_latest_eval_script()
                self.agents_dict['test_analysis_agent'].dockerfile = dockerfile
                self.agents_dict['test_analysis_agent'].eval_script_skeleton = eval_script_skeleton
                self.agents_dict['test_analysis_agent'].eval_script = eval_script
                # analysis, _, success =  self.agents_dict['test_analysis_agent'].run_task()
              
                if self.disable_run_test:
                    
                    analysis, _, success =  self.agents_dict['test_analysis_agent'].run_task_without_run_test()
                else:
                    analysis, _, success =  self.agents_dict['test_analysis_agent'].run_task(self.disable_context_retrieval)
                self.dump_cost()
                if isinstance(analysis, str):
                    try:
                        analysis = json.loads(analysis)
                    except:
                        analysis = {}
                else:
                    analysis = {}


                is_finish = analysis.get("is_finish", None)
                if is_finish:
                    self.workflow_finish_status = True
                    break
                
                # write dockerile + eval script + build contaier + run eval script
                # collect feedback (image error + test error)
                # image error: 1. modify dockerfile directly
                #              2. go to context retrieval  agent for more information.
                # test error: 1. go to modfiy dockerfile.
                #             2. or go to collect more information

                # scheduler
                guidance_for_context_retrieval_agent = analysis.get("guidance_for_context_retrieval_agent", None)
                if guidance_for_context_retrieval_agent:
                    self.set_agent_status("context_retrieval_agent",False)
                    self.agents_dict['context_retrieval_agent'].add_user_message(f'After setting up dockerfile and running tests, the test log analysis agent find that there is other context information need to collect. Here is his analysis:\n{guidance_for_context_retrieval_agent}\n\n')

                guidance_for_write_dockerfile_agent = analysis.get("guidance_for_write_dockerfile_agent", None)
                if guidance_for_write_dockerfile_agent:
                    self.set_agent_status("write_docker_agent",False)
                    self.agents_dict['write_docker_agent'].add_user_message(f'After setting up dockerfile and running tests, the test log analysis agent find that there is a problem with dockefile. Here is his analysis:\n{guidance_for_write_dockerfile_agent}\n\n')

                guidance_for_write_eval_script_agent = analysis.get("guidance_for_write_eval_script_agent", None)
                if guidance_for_write_eval_script_agent:
                    self.set_agent_status("write_eval_script_agent",False)
                    self.agents_dict['write_eval_script_agent'].add_user_message(f'After setting up dockerfile and running tests, the test log analysis agent find that there is a problem with eval script. Here is his analysis:\n{guidance_for_write_eval_script_agent}\n\n')

        else:
            log_msg = "Exceed largest number of tries.."
            logger.info(f"Too many rounds. {log_msg}")

        dockerfile_content = self.agents_dict['write_docker_agent'].get_latest_dockerfile()
        eval_script_content = self.agents_dict['write_eval_script_agent'].get_latest_eval_script()
        eval_script_skeleton_content = self.agents_dict['write_eval_script_agent'].get_latest_eval_script_skeleton()
        if dockerfile_content and eval_script_content:
            dockerfile_path = os.path.join(self.output_dir, "Dockerfile")
            logger.info(f"Writing Dockerfile to: {dockerfile_path}")
            with open(dockerfile_path, "w") as dockerfile_f:
                dockerfile_f.write(dockerfile_content)

        
            eval_script_path = os.path.join(self.output_dir, "eval.sh")
            logger.info(f"Writing eval.sh to: {eval_script_path}")
            with open(eval_script_path, "w") as eval_script_f:
                eval_script_f.write(eval_script_content)


        status_file_path = os.path.join(self.output_dir, "status.json")
        logger.info(f"Writing status.json to: {status_file_path}")
        with open(status_file_path, "w") as status_file_f:
                json.dump({"is_finish": self.workflow_finish_status}, status_file_f)

        if self.workflow_finish_status:
            recs = self._read_results()
            info = deepcopy(self.task.task_info)

            # merge in your new fields
            info.update({
                "dockerfile": dockerfile_content,
                "eval_script": eval_script_content,
                "eval_script_skeleton": eval_script_skeleton_content,
                # keep any other existing keys from task_info
            })

            recs.append(info)
            self._write_results(recs)
        