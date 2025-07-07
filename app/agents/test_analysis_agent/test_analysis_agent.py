from pathlib import Path
import os
from loguru import logger
from app.agents.agent import Agent
from app.data_structures import FunctionCallIntent, MessageThread
from app.task import Task
from app.agents.test_analysis_agent import test_analysis_utils
from app.agents.test_analysis_agent.docker_utils  import (
    cleanup_container,
    remove_image,
    copy_to_container,
    exec_run_with_timeout,
    BuildImageError,
    build_container,
    EvaluationError)
import docker
import re
from app.log import log_exception,setup_logger,close_logger
from app.log import (
    print_acr,
    print_banner,
    print_retrieval,
)
import json
from os.path import join as pjoin
import traceback
from typing import Optional

MAX_LINE_NUM = 600
ansi_escape = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")

def sanitize_docker_image_name(name: str) -> str:
    if not name or not str(name).strip():
        return "swe-task"
    sanitized = re.sub(r'[^a-z0-9_-]', '-', str(name).lower().strip())
    if sanitized.startswith('-'):
        sanitized = 'swe' + sanitized
    # Remove trailing hyphens/underscores, then leading underscores
    sanitized = sanitized.rstrip('-').rstrip('_').lstrip('_')
    if not sanitized or sanitized == "swe":
        return "swe-task"
    if len(sanitized) > 50:
        sanitized = sanitized[:50]
    return sanitized

class TestAnalysisAgent(Agent):
    """
    Agent responsible for:
      1. Loading the latest test_output.txt
      2. Formatting it with line numbers and truncation
      3. Sending it to the test-log-analysis utility (agent_analyze_test_log)
    """
    api_functions = ["setup_docker_and_run_test"]

    def __init__(self, task: Task, output_dir: str, repo_basic_info: str, client:docker.DockerClient):
        super().__init__(agent_id=self.__class__.__name__)
        self.msg_thread  = MessageThread()
        self.task = task
        self.output_dir = os.path.abspath(output_dir)
        self.analysis_count = 0
        self.run_test_num = 0
        self.setup_dockerfile_num = 0
        self.repo_basic_info = repo_basic_info
        # Sanitize task_id to ensure valid Docker image names
        raw_task_id = getattr(task, 'task_id', '') or ''
        self.task_id = sanitize_docker_image_name(raw_task_id)
        self.client = client
        self.test_analysis_dir = os.path.join(self.output_dir, "test_analysis_agent") 
        # self.build_image_dir = os.path.join(self.output_dir, "build_image") 
        # self.run_test_dir = os.path.join(self.output_dir, "run_test") 
        self.eval_script_skeleton = None
        self.dockerfile = None
        self.eval_script = None
        self.timeout = 3600
        self.disable_context_retrieval = False
        self.disable_run_test = False
        # self.init_msg_thread()



    def init_msg_thread(self) -> None:
        """
        Reset the message thread and inject the base prompt before each run_task.
        """
        self.msg_thread = MessageThread()
        # Choose the appropriate system prompt
        # if getattr(agent_analyze_test_log, "SYSTEM_PROMPT_WIT_WEB_SEARCH", None) and getattr(self.task, 'enable_web_search', False):
        #     self.msg_thread.add_system(test_analysis_utils.SYSTEM_PROMPT_WIT_WEB_SEARCH)
        # else:
        if self.disable_context_retrieval:
            self.add_system_message(test_analysis_utils.SYSTEM_PROMPT_WITHOUT_CONTEXT_RETRIEVAL)
        elif self.disable_run_test:
            self.add_system_message(test_analysis_utils.SYSTEM_PROMPT_WITHOUT_RUN_TEST)
        else:
            self.add_system_message(test_analysis_utils.SYSTEM_PROMPT)
        # Inject repository basic information
        self.add_user_message(self.repo_basic_info)
        self.add_user_message(f'The current dockerfile used to setup environemnt:\n{self.dockerfile}')
        self.add_user_message(f'The current eval script (omit test patch to decrease length) used to run tests:\n{self.eval_script_skeleton}')

    def get_latest_test_analysis_output_dir(self):
        output_dir = f'{self.test_analysis_dir}_{self.analysis_count}'
        return output_dir

    def get_latest_test_log(self) -> str:
        """Read the latest test_output.txt produced by run_test."""
        test_dir = self.get_latest_test_analysis_output_dir()
        path = os.path.join(test_dir, "test_output.txt")
        try:
            return Path(path).read_text()
        except FileNotFoundError:
            return ""
    

    def get_test_log_with_line_numbers(self) -> str:
        test_log = self.get_latest_test_log()
        lines = test_log.splitlines()
        
       
        width = len(str(len(lines)))
        full_formatted = [f"{i + 1:>{width}}   {line}" for i, line in enumerate(lines)]
        
        if len(full_formatted) <= MAX_LINE_NUM:
            return f'Test log:\n{chr(10).join(full_formatted)}\n\n'
        
        head_size = MAX_LINE_NUM // 2
        tail_size = MAX_LINE_NUM - head_size
        
        head = full_formatted[:head_size]
        tail = full_formatted[-tail_size:]
        
       
        omission = " " * width + "   [..., {} lines omitted ...]".format(
            len(full_formatted) - head_size - tail_size)
        
        truncated_log = chr(10).join(head + [omission] + tail)
        
        return f'Test log (showing first {head_size} & last {tail_size} lines):\n{truncated_log}\n\n'

    def run_task(self, disable_context_retrieval= False, print_callback=None) -> tuple[str, str, bool]:
        """
        2. Read and format the test log
        3. Add formatted log to the message thread
        4. Invoke agent_analyze_test_log.run_with_retries(...)
        5. Return (tool_output, summary, ok)
        """
        self.init_msg_thread()
        print_banner(f"Task {self.task.task_id} Iteration ROUND {self.iteration_num} Try to setup docker and run tests ")
        
        self.analysis_count += 1
        test_log_output_dir = self.get_latest_test_analysis_output_dir()
        os.makedirs(test_log_output_dir,exist_ok=True)
        intent = FunctionCallIntent("setup_docker_and_run_test", {}, None)
        tool_output, _, success = self.dispatch_intent(intent)
        build_image_status = False
        if 'Image built successfully!' not in tool_output:
            # fail to build image. go to dockefile refine.
            print_acr(
                'Build Image Failure!',
                f"Task {self.task.task_id} Iteration ROUND {self.iteration_num}  test analysis",
                print_callback=print_callback,
            )
            error_in_building_dockerfile = f'We can not run tests successfully, cause we encounter some errors when building dockerfile. As follows:\n{tool_output}\n\n'
            self.add_user_message(error_in_building_dockerfile)
        elif success:
            build_image_status  = True
            print_acr(
                'Build Image Successfully!',
                f"Task {self.task.task_id} Iteration ROUND {self.iteration_num}  test analysis ",
                print_callback=print_callback,
            )
            test_log = self.get_test_log_with_line_numbers()
            # self.add_user_message(f'Eval script (We omit details of test patch):\n{self.eval_script_skeleton}\n\n')
            self.add_user_message(test_log)
        else:
            logger.error(tool_output)
            logger.error('some problem in running tests')
            return None, f'{self.agent_id} fails, somt error happens', False
        # if we judge that we achieve the goal, terminate the process
        # if test log show that it fails, we go to plan for futrure directions
        # judge whether achieve the goal, if not planning for the work in the next stage.
        print_acr(
                f'Task {self.task.task_id} Iteration ROUND {self.iteration_num}  Try to analyze the test log ',
                f"Task {self.task.task_id} Iteration ROUND {self.iteration_num}  test analysis round ",
                print_callback=print_callback,
            )
            
        success =False
        analysis = test_analysis_utils.run_with_retries(self.msg_thread,disable_context_retrieval=disable_context_retrieval,print_callback=print_callback)
        task_output = analysis
        analysis_file = Path(f"{self.get_latest_test_analysis_output_dir()}/analysis.json")

        to_save = {}
        if isinstance(analysis, dict):
            to_save = analysis
        elif isinstance(analysis, str):
            try:
                to_save = json.loads(analysis)
            except Exception as e:
                to_save = {}
        else:
            # analysis 既不是 dict 也不是 str，按需处理
            to_save = {}

        to_save['build_image_status'] = build_image_status

        if task_output is None:
            summary = "The tool returned nothing. The main agent probably did not provide enough clues."
            success = False
          
            
        else:
            summary = "The tool returned the selected search APIs in json format generated by another agent."
            success = True
           
            
        with analysis_file.open("w", encoding="utf-8") as f:
            json.dump(to_save, f, ensure_ascii=False, indent=2)
        
        #need to save analysis into json file
        conversation_file = pjoin(test_log_output_dir, f"conversation.json")
        self.msg_thread.save_to_file(conversation_file)
        
        return task_output, summary, success 

    
    def run_task_without_run_test(self, print_callback=None) -> tuple[str, str, bool]:
        """
        This function is just for ablation study
        """
        self.init_msg_thread()
        print_banner(f"Task {self.task.task_id} Iteration ROUND {self.iteration_num} Try to setup docker and run tests ")
        
        self.analysis_count += 1
        test_log_output_dir = self.get_latest_test_analysis_output_dir()
        os.makedirs(test_log_output_dir,exist_ok=True)
       
        # if we judge that we achieve the goal, terminate the process
        # if test log show that it fails, we go to plan for futrure directions
        # judge whether achieve the goal, if not planning for the work in the next stage.
        print_acr(
                f'Task {self.task.task_id} Iteration ROUND {self.iteration_num}  Try to analyze the test log ',
                f"Task {self.task.task_id} Iteration ROUND {self.iteration_num}  test analysis round ",
                print_callback=print_callback,
            )
            
        success =False
        analysis = test_analysis_utils.run_with_retries(self.msg_thread,disable_run_test=True,print_callback=print_callback)
        task_output = analysis
        analysis_file = Path(f"{self.get_latest_test_analysis_output_dir()}/analysis.json")

        to_save = {}
        if isinstance(analysis, dict):
            to_save = analysis
        elif isinstance(analysis, str):
            try:
                to_save = json.loads(analysis)
            except Exception as e:
                to_save = {}
        else:
            # analysis 既不是 dict 也不是 str，按需处理
            to_save = {}

        # to_save['build_image_status'] = build_image_status

        if task_output is None:
            summary = "The tool returned nothing. The main agent probably did not provide enough clues."
            success = False
          
            
        else:
            summary = "The tool returned the selected search APIs in json format generated by another agent."
            success = True
           
            
        with analysis_file.open("w", encoding="utf-8") as f:
            json.dump(to_save, f, ensure_ascii=False, indent=2)
        
        #need to save analysis into json file
        conversation_file = pjoin(test_log_output_dir, f"conversation.json")
        self.msg_thread.save_to_file(conversation_file)
        
        return task_output, summary, success 
       

    def build_docker_image(
        self,
        dockerfile,
        cur_build_image_dir,
        task_id,
        image_name,
        build_image_logger,
        client
    ):
        """Build Docker image with detailed logging and error handling."""
    
        
        build_image_logger.info(
            f"Building image {task_id}\n"
            f"Using dockerfile:\n{dockerfile}\n"
        )

    

        if self.setup_dockerfile_num > 1:
            # prev_image_name = f"{task_id}:latest_{setup_dockerfile_num - 1}"
            prev_image_name = f"{self.task_id}-dockerfile{self.setup_dockerfile_num-1}:latest"
            try:
                client.images.remove(prev_image_name, force=True)
                build_image_logger.info(f"Deleted previous image: {prev_image_name}")

            except docker.errors.ImageNotFound:
                build_image_logger.info(f"Do not find previous image, images list is clean.")
            except Exception as e: 
                build_image_logger.error(f"Failed to delete previous image {prev_image_name}: {str(e)}")
        
        

        dockerfile_path = f'{cur_build_image_dir}/Dockerfile'
        logger.info(f"Writing Dockerfile to: {dockerfile_path}")
        with open(dockerfile_path, "w") as f:
            f.write(dockerfile)

        
        command_output = []  
        capturing = False   
        response = client.api.build(
            path=cur_build_image_dir,
            tag=image_name,
            rm=True,
            forcerm=True,
            decode=True,
            platform="linux/x86_64",
            nocache=True,
        )

        buffer = ""

       
        for chunk in response:
            if "stream" in chunk:
              
                buffer += ansi_escape.sub("", chunk["stream"])
                
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    if not line.strip():
                        continue

                    
                    if line.startswith("Step "):
                        last_command = line
                        command_output = [line]
                        capturing = True
                    elif capturing:
                        command_output.append(line)

                 
                    build_image_logger.info(line)

            elif "errorDetail" in chunk and capturing:
               
                if buffer.strip():
                    command_output.append(buffer.strip())
                    build_image_logger.info(buffer.strip())
                    buffer = ""

             
                error_msg = ansi_escape.sub("", chunk["errorDetail"]["message"])
                build_image_logger.error(f"Error: {error_msg}")
                command_output.append(f"Error: {error_msg}")

             
                raise docker.errors.BuildError(error_msg, build_log=command_output)

      
        if buffer.strip():
            build_image_logger.info(buffer.strip())

        build_image_logger.info("Image built successfully!")
    def setup_docker_and_run_test(
        self
    ) -> tuple[str, str, bool]:
        # building docker image first
       
        dockerfile = self.dockerfile
        
        eval_script = self.eval_script
        tool_output = ""
        summary = ""
        success = False
        self.setup_dockerfile_num += 1
        cur_build_image_dir = self.get_latest_test_analysis_output_dir()
        os.makedirs(cur_build_image_dir, exist_ok=True)
        build_image_logger = setup_logger(self.task_id, Path(f'{cur_build_image_dir}/build_image.log'))
        # Generate a valid Docker image name
        image_name = f"{self.task_id}-dockerfile{self.setup_dockerfile_num}:latest"
       
        try:
            self.build_docker_image(dockerfile,
                                    cur_build_image_dir,
                                   
                                    self.task_id, 
                                    image_name,
                                    build_image_logger,
                                    self.client) 
            tool_output += "Image built successfully!\n"
            summary += f"Docker image {image_name} built successfully.\n"
        except docker.errors.BuildError as e:
           
            build_log = e.build_log
            if len(build_log) > MAX_LINE_NUM:
                half = MAX_LINE_NUM // 2
                skipped = len(build_log) - MAX_LINE_NUM
                build_log = (
                    build_log[:half]
                    + [f"...skipped {skipped} lines..."]
                    + build_log[-half:]
                )
            tool_output += "\n".join(build_log)
            build_image_logger.error(e)
            summary += f"Failed to build Docker image."
            success = False
            return tool_output, summary, success
        except Exception as e:
          
            build_image_logger.error(f"Unexpected error: {str(e)}")
            tool_output += f'{str(e)}\n'
            summary += f"Unexpected error when building images."
            success = False
            return tool_output, summary, success
        finally:
            close_logger(build_image_logger)

        test_output, test_summary, test_success = self.run_test(eval_script)
        tool_output += test_output
        summary += test_summary
        success = test_success

        return tool_output, summary, success

    def run_test(self, eval_script: str) -> (str, str, bool):
        tool_output = ""
        summary = ""
        success = False
        patch = self.task.patch
        self.run_test_num += 1
        self.reset_tool_sequence()
        cur_test_dir = self.get_latest_test_analysis_output_dir()
        os.makedirs(cur_test_dir, exist_ok=True)
        run_test_logger = setup_logger(self.task_id, Path(f'{cur_test_dir}/run_test.log'))
        # Use sanitized image names to ensure valid Docker tags
        test_image_name = f"{self.task_id}-dockerfile{self.setup_dockerfile_num}:latest"
        test_container_name = f"{self.task_id}-test{self.run_test_num}"
        instance_id = self.task_id
        container = None
        test_output_path = f'{cur_test_dir}/test_output.txt'
        try:
            container = build_container(self.client,test_image_name,test_container_name,instance_id,run_test_logger)

            container.start()
            run_test_logger.info(f"Container for {instance_id} started: {container.id}")
            tool_output += f"Container {container.id} started.\n"
            summary += "Container started.\n"
            # Copy model prediction as patch file to container
            patch_file = Path(f"{cur_test_dir}/patch.diff")
            patch_file.write_text(patch or "")
            run_test_logger.info(
                f"Intermediate patch for {instance_id} written to {patch_file}, now applying to container..."
            )
            copy_to_container(container, patch_file, Path("/tmp/patch.diff"))

        
            # Attempt to apply patch to container
            val = container.exec_run(
                "git apply -p1 -v /tmp/patch.diff",
                workdir="/testbed",
                user="root",
            )
            if val.exit_code != 0:
                run_test_logger.info(f"Failed to apply patch to container, trying again...")
                
                # try "patch --batch --fuzz=5 -p1 -i {patch_path}" to try again
                val = container.exec_run(
                    "patch --batch --fuzz=5 -p1 -i /tmp/patch.diff",
                    workdir="/testbed",
                    user="root",
                )
                if val.exit_code != 0:
                    run_test_logger.info(f"Apply patch fail:\n{val.output.decode('utf-8')}")
                    raise EvaluationError(
                        instance_id,
                        f"Apply patch fail:\n{val.output.decode('utf-8')}. Check if you apply patch in incorrect directories.",
                        run_test_logger,
                    )
                else:
                    run_test_logger.info(f"Apply patch success:\n{val.output.decode('utf-8')}")
            else:
                run_test_logger.info(f"Apply patch success:\n{val.output.decode('utf-8')}")
            tool_output += "Patch applied successfully.\n"
            summary += "Patch applied.\n"
                    # Get git diff before running eval script
            git_diff_output_before = (
                container.exec_run("git diff", workdir="/testbed").output.decode("utf-8").strip()
            )
            run_test_logger.info(f"Git diff before:\n{git_diff_output_before}")

            eval_file = Path(f"{self.get_latest_test_analysis_output_dir()}/eval.sh")
            logger.info(f"Writing eval.sh to: {eval_file}")
            eval_file.write_text(eval_script)
            run_test_logger.info(
                f"Eval script for {instance_id} written to {eval_file}, now applying to container..."
            )
            copy_to_container(container, eval_file, Path("/eval.sh"))

            # Run eval script, write output to logs
            result = exec_run_with_timeout(container, "/bin/bash /eval.sh", timeout=self.timeout)
            test_output = result.decode("utf-8")
            
            with open(test_output_path, "w") as f:
                f.write(test_output)
            run_test_logger.info(f"Test output for {instance_id} written to {test_output_path}")

            # Get git diff after running eval script
            git_diff_output_after = (
                container.exec_run("git diff", workdir="/testbed").output.decode("utf-8").strip()
            )

            # Check if git diff changed after running eval script
            run_test_logger.info(f"Git diff after:\n{git_diff_output_after}")
            if git_diff_output_after != git_diff_output_before:
                run_test_logger.info(f"Git diff changed after running eval script")
                tool_output += "Note: Git diff changed after test execution.\n"
                summary += "Git diff changed.\n"

        except EvaluationError as e:
            error_msg = (f"EvaluationError {instance_id}: {e}\n"
                        f"{traceback.format_exc()}\n"
                        f"Check ({run_test_logger.log_file}) for more information.")
            run_test_logger.info(error_msg)
            tool_output += error_msg + "\n"
            summary += "Evaluation error occurred.\n"
            success = False
           
        except Exception as e:
            error_msg = (f"Error in evaluating model for {instance_id}: {e}\n"
                        f"{traceback.format_exc()}\n"
                        f"Check ({run_test_logger.log_file}) for more information.")
            run_test_logger.info(error_msg)
            tool_output += error_msg + "\n"
            summary += "Unexpected error occurred.\n"
            success = False
        else:
            if not os.path.exists(test_output_path):
                tool_output += "Do not generate test_output.txt. Please check the correctness of dockerfile and eval script.\n"
                summary += 'Fail to obtain test results.'
                success = False
            else:
                tool_output += f"Find test_output.txt! Waiting for analysis. "
                summary += 'Obtain test results successfully.'
                success = True

        finally:
           
            # Remove instance container + image, close logger
            cleanup_container(self.client, container,run_test_logger)
            
            remove_image(self.client, test_image_name, run_test_logger)
            close_logger(run_test_logger)
        self.dump_tool_sequence(self.get_latest_test_analysis_output_dir())
        return tool_output, summary, success