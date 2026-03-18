# agents/docker_manager.py
import os
import uuid
import shutil
import tempfile

try:
    import docker
    from docker.errors import ContainerError, APIError, DockerException
except Exception:
    docker = None
    DockerException = Exception
    ContainerError = Exception
    APIError = Exception


class DockerManager:
    """
    Lazy-initialized Docker client. If Docker is not available or disabled,
    run_code() raises a clear exception that ToolAgent can translate to a user-friendly message.
    """

    def __init__(self):
        self._client = None
        self._disabled = os.getenv("DISABLE_DOCKER", "0") == "1"

    def _ensure_client(self):
        if self._disabled:
            raise DockerException("Docker is disabled via DISABLE_DOCKER=1")

        if docker is None:
            raise DockerException("python 'docker' package not installed")

        if self._client is None:
            # Will raise if Docker Desktop is not running / pipe not available
            self._client = docker.from_env()
            # Touch the daemon to confirm connectivity
            _ = self._client.version()

        return self._client

    def run_code(self, code: str, timeout: int = 30) -> str:
        client = self._ensure_client()

        image = "python:3.11-slim"
        container_name = f"code_runner_{uuid.uuid4().hex[:8]}"

        # use a temp dir that we can always clean
        workdir = tempfile.mkdtemp(prefix=container_name + "_")
        script_path = os.path.join(workdir, "script.py")

        with open(script_path, "w", encoding="utf-8") as f:
            f.write(code)

        try:
            # Pull once if needed
            try:
                client.images.pull(image)
            except Exception:
                # ignore pull errors; may already exist or policy blocked
                pass

            result = client.containers.run(
                image=image,
                name=container_name,
                command=["python", "script.py"],
                volumes={workdir: {"bind": "/workspace", "mode": "ro"}},
                working_dir="/workspace",
                detach=False,
                remove=True,
                stderr=True,
                stdout=True,
                mem_limit="256m",
                nano_cpus=500_000_000,
                network_mode="none",     # default: NO network for safety
                # dns=["8.8.8.8"],       # only if you later enable network_mode="bridge"
                environment={
                    # keep empty by default; add proxies only if needed
                },
                runtime=None
            )
            return result.decode("utf-8")
        except ContainerError as e:
            try:
                return f"Execution error:\n{e.stderr.decode('utf-8')}"
            except Exception:
                return f"Execution error: {str(e)}"
        except (APIError, DockerException) as e:
            raise DockerException(f"Docker error: {e}")
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
