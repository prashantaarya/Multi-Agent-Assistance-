# agents/docker_manager.py
import docker
from docker.errors import ContainerError, APIError
import uuid
import shutil
import os

class DockerManager:
    def __init__(self):
        self.client = docker.from_env()

    def run_code(self, code: str, timeout: int = 30):
        """Run the given Python code in a fresh container."""
        image = "python:3.11-slim"
        container_name = f"code_runner_{uuid.uuid4().hex[:8]}"
        workdir = f"/tmp/{container_name}"
        os.makedirs(workdir, exist_ok=True)

        script_path = os.path.join(workdir, "script.py")
        with open(script_path, "w") as f:
            f.write(code)

        try:
            result = self.client.containers.run(
                image=image,
                name=container_name,
                command=["python", "script.py"],
                volumes={ workdir: {"bind": "/workspace", "mode": "ro"} },
                working_dir="/workspace",
                detach=False,
                remove=True,
                stderr=True,
                stdout=True,
                mem_limit="256m",
                nano_cpus=500_000_000,
                network_mode="bridge",  # or just remove network_disabled
                dns=["8.8.8.8"],        # ✅ Fix DNS resolution
                environment={
                    "HTTP_PROXY": "http://host.docker.internal:3128",
                    "HTTPS_PROXY": "http://host.docker.internal:3128"
                },
                timeout=timeout
            )
            
            
            return result.decode("utf-8")
        except ContainerError as e:
            return f"Execution error: {e.stderr.decode('utf-8')}"
        except APIError as e:
            return f"Docker API error: {str(e)}"
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
