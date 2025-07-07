from __future__ import annotations

import os
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from os.path import join as pjoin
from tempfile import mkstemp
import shutil
import app.utils as apputils
from app import globals, log
from app import utils as app_utils

from app.log import log_and_print
from docker import DockerClient

class Task(ABC):
    @property
    @abstractmethod
    def project_path(self) -> str:
        raise NotImplementedError("abstract method")

    @abstractmethod
    def get_issue_statement(self) -> str:
        raise NotImplementedError("abstract method")

    @abstractmethod
    def setup_project(self) -> None:
        """Set up the project before starting to resolve the task."""
        raise NotImplementedError("abstract method")

    @abstractmethod
    def reset_project(self) -> None:
        """Reset project to initial state."""
        raise NotImplementedError("abstract method")



@dataclass(kw_only=True)
class SweTask(Task):
    task_id: str
    problem_statement: str
    repo_path: str
    repo_cache_path: str
    commit: str
    # env_name: str
    repo_name: str
    # pre_install_cmds: list[str]
    # install_cmd: str
    # test_cmd: str
    patch: str
    test_patch: str
    # testcases_passing: list[str]
    # testcases_failing: list[str]
    language: str
    # image_urls: list[str]
    # reference_setup: dict
    version: str
    client: DockerClient
    task_info: dict
    @property
    def project_path(self) -> str:
        return self.repo_path
    

    @project_path.setter
    def project_path(self, value: str) -> None:
        self.repo_path = value

    def get_issue_statement(self) -> str:
        return self.problem_statement
    

    def setup_project(self) -> None:
        # get the correct version of the project and commit-specific pip install
        task = self
        with apputils.cd(task.project_path):
            apputils.repo_reset_and_clean_checkout(task.commit)


        # apply the test modifications to this task

        # commit the current changes, so that resetting later do not erase them
        with apputils.cd(task.project_path):
            apputils.repo_commit_current_changes()

    def reset_project(self) -> None:
        with apputils.cd(self.repo_path):
            apputils.repo_reset_and_clean_checkout(self.commit)

    def remove_project(self) -> None:
        """Remove the entire project repository."""
        if os.path.exists(self.repo_path):
            shutil.rmtree(self.repo_path)
            log_and_print(f"Removed project repository at {self.repo_path}")





@dataclass(kw_only=True)
class PlainTask(Task):
    """
    Tasks that only contain a codebase and an issue descripion (no test suite).
    """

    commit_hash: str
    local_path: str
    problem_statement: str
    patch: str = ""
    test_patch: str = ""
    repo_name: str = ""
    version: str = ""
    task_id: str = ""

    @property
    def commit(self) -> str:
        return self.commit_hash

    @property
    def project_path(self) -> str:
        return self.local_path

    @project_path.setter
    def project_path(self, value: str) -> None:
        self.local_path = value

    def setup_project(self) -> None:
        with apputils.cd(self.project_path):
            apputils.repo_reset_and_clean_checkout(self.commit_hash)

    def reset_project(self) -> None:
        with apputils.cd(self.project_path):
            apputils.repo_reset_and_clean_checkout(self.commit_hash)

    def get_issue_statement(self) -> str:
        return self.problem_statement


