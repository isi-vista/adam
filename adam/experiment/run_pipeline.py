"""
A script intended to run the entire Phase 3 pipeline (as of milestone 6).

Note that this script does not take care of starting the object segmentation server. You have to do
that separately.

Note also that this writes to then reads from a params file for ADAM. If you run this again with the
same params file name while a job is running, you will overwrite the old parameters! So be careful.
"""
import atexit
from datetime import datetime
from io import StringIO
from locale import getpreferredencoding
import logging
import os
from os import getenv
from pathlib import Path
import shlex
from shutil import copytree
from subprocess import run, CalledProcessError
from typing import List, Mapping, NewType, Optional, Sequence

import yaml

from vistautils.parameters_only_entrypoint import parameters_only_entry_point
from vistautils.parameters import Parameters, ParameterError

logger = logging.getLogger(__name__)


SlurmJobID = NewType("SlurmJobID", str)
Email = NewType("Email", str)
MailType = NewType("MailType", str)

slurm_jobs_scheduled: List[SlurmJobID] = []


def stop_slurm_jobs():
    """
    Exit handler to clean up Slurm jobs.
    """
    for job_id in slurm_jobs_scheduled:
        logger.info("Canceling previously scheduled job %s...", job_id)
        cancel_command = ["scancel", str(job_id)]
        print(" ".join(shlex.quote(part) for part in cancel_command))
        run(cancel_command, check=False)


def build_sbatch_command(
    script_path: Path,
    *,
    dependencies: Sequence[SlurmJobID] = (),
    job_name: Optional[str] = None,
    log_dir: Optional[Path] = None,
    email: Optional[Email] = None,
    mail_types: Sequence[MailType] = (),
    extra_sbatch_args: Sequence[str] = (),
    script_args: Sequence[str],
) -> Sequence[str]:
    """
    Construct a Slurm command to run the given sbatch script in a parseable way.

    This returns a sequence of commandline arguments that can be passed directly to
    subprocess.run(). This function passes --parsable to sbatch to force the output to an easily
    machine-readable format; note however that even with this flag errors are printed in the usual
    way.

    Parameters:
        script_path: The path to the Slurm/sbatch script to run.
        dependencies:
            A sequence of Slurm job IDs this job should depend on. This job will only run after
            every dependency has completed successfully.
        job_name: A custom name to use for the job.
        log_dir: A directory in which to save log files.
        email: An email to use for mail notifications.
        mail_types: A list of Slurm mail types to specify.
        extra_sbatch_args: Sequence of args to be passed to sbatch, if any.
        script_args: Arguments to pass to the script.
    """
    result = ["sbatch", "--parsable", "--requeue"]
    if dependencies:
        result.extend(
            [
                # Don't run this job until the dependencies complete successfully.
                f"--dependency=afterok:{':'.join(dependencies)}",
                # If any of the dependencies fail, cancel this job.
                "--kill-on-invalid-dep=yes",
            ]
        )
    if job_name:
        result.append(f"--job-name={job_name}")
    if log_dir:
        log_dir.mkdir(exist_ok=True, parents=True)
        result.append(f"--output={str(log_dir.joinpath('R-%x.%j.out'))}")

    result.extend(_build_email_arg(email=email, mail_types=mail_types))
    result.extend(extra_sbatch_args)
    result.append(str(script_path))
    result.extend(script_args)

    return result


def build_bash_command(
    script_path: Path,
    *,
    # pylint:disable=unused-argument
    dependencies: Sequence[SlurmJobID] = (),
    job_name: Optional[str] = None,
    log_dir: Optional[Path] = None,
    email: Optional[Email] = None,
    mail_types: Sequence[MailType] = (),
    extra_sbatch_args: Sequence[str] = (),
    # pylint:enable=unused-argument
    script_args: Sequence[str],
) -> Sequence[str]:
    """
    Construct a Bash command to run the given sbatch script.

    This returns a sequence of commandline arguments that can be passed directly to
    subprocess.run(). Note the sbatch commands are ignored.

    Parameters:
        script_path: The path to the Slurm/sbatch script to run.
        dependencies:
            A sequence of Slurm job IDs this job should depend on. These are ignored.
        job_name: A custom name to use for the job.
        log_dir: A directory in which to save log files. Ignored.
        email: An email to use for mail notifications.
        mail_types: A list of Slurm mail types to specify.
        extra_sbatch_args: Sequence of args to be passed to sbatch, if any.
        script_args: Arguments to pass to the script.
    """
    result = ["bash", str(script_path)]
    result.extend(script_args)
    return result


def _build_email_arg(
    *,
    email: Optional[str] = None,
    mail_types: Sequence[str] = (),
) -> Sequence[str]:
    result = []
    if email and mail_types:
        result = [f"--mail-user={email}", f"--mail-type={','.join(mail_types)}"]
    return result


def echo_command(command: Sequence[str], *, save_to: Optional[Path]) -> Sequence[str]:
    """
    Echo the given command returning it unchanged.

    If passed, save_to should be the path to a file. When passed the command will also append the
    output to the given file.
    """
    output = " ".join(shlex.quote(part) for part in command)
    print(output)
    if save_to is not None:
        with save_to.open(mode="a", encoding="utf-8") as save_to_file:
            print(output, file=save_to_file)
    return command


def run_slurm(
    command: Sequence[str], *, env: Mapping[str, str], workdir: Path
) -> Optional[SlurmJobID]:
    """
    Run a constructed Slurm command, returning the job ID on success and None otherwise.

    Parameters:
        command: The pre-constructed Slurm command.
        env: Environment variables to set, if any.
    """
    try:
        completed_process = run(
            command, cwd=workdir, capture_output=True, env=env, check=True
        )
        result = SlurmJobID(
            completed_process.stdout.decode(getpreferredencoding()).strip()
        )
        slurm_jobs_scheduled.append(result)
    except CalledProcessError as e:
        raise ValueError(f"Job run failed for command {command}") from e
    return result


def run_bash(
    command: Sequence[str], *, env: Mapping[str, str], workdir: Path
) -> Optional[SlurmJobID]:
    """
    Run a constructed Bash command, returning None because Bash doesn't produce Slurm job IDs.

    This is meant for debugging. As a result it always runs the command synchronously. It also pipes
    the output so you can see what is going on.

    Parameters:
        command: The pre-constructed Slurm command.
        env: Environment variables to set, if any.
    """
    run(command, cwd=workdir, env=env, check=True)
    return None


def dependency_list(
    *dependencies: Optional[SlurmJobID],
) -> Sequence[SlurmJobID]:
    """
    Build a dependency list, dropping any null dependencies.
    """
    return tuple(dependency for dependency in dependencies if dependency is not None)


def add_to_path(bin_dir: Path) -> str:
    """
    Construct a new path string by prepending bin_dir to the current PATH.
    """
    path_elements = [str(bin_dir)]

    existing_path = getenv("PATH")
    if existing_path:
        path_elements.append(existing_path)

    return ":".join(path_elements)


def parse_bool_param(params: Parameters, param_name: str) -> bool:
    """
    Parse a boolean parameter from the given parameters.

    This is a workaround due to how vistautils handles -p parameters. -p parameters are handy for
    debugging. However, the typed parameter methods don't interact well with them. Specifically,
    when you specify a key-value pair with -p, the value is not parsed as YAML, so `-p key true`
    results in a string value `"true"` for the key `key`. Out of the various parameter types we may
    want to change during debugging, booleans and paths are probably the most useful, and paths
    are safe because they have string values under the hood. Hence, this workaround.
    """
    try:
        result = params.boolean(param_name)
    except ParameterError:
        result = yaml.safe_load(StringIO(params.string(param_name)))
    return result


def is_empty_dir(path: Path) -> bool:
    """
    Return True if the given path is an empty directory, otherwise false.
    """
    child = next(path.iterdir(), None)
    return path.is_dir() and child is None


def ignore_from_base_curriculum(_parent: str, children: Sequence[str]) -> Sequence[str]:
    def is_raw_input_file(path_str: str) -> bool:
        path = Path(path_str)
        return (
            path.name.startswith("semantic")
            or path.name.startswith("original_colors_color_segmentation_")
            or path.name.startswith("color_segmentation_")
            or path.name.startswith("color_refined_semantic_")
            or path.name.startswith("combined_color_refined_semantic_")
            or path.name.startswith("stroke_")
            or path.name.startswith("feature")
            or path.name.startswith("post_decode")
        )

    return [child for child in children if is_raw_input_file(child)]


def make_run_identifier(run_start: datetime) -> str:
    return run_start.strftime("%Y-%m-%d_%H:%M:%S%z")


def pipeline_entrypoint(params: Parameters) -> None:
    root = params.existing_directory("adam_root")

    pipeline_params = params.namespace("pipeline")
    use_sbatch = parse_bool_param(pipeline_params, "use_sbatch")
    do_object_segmentation = parse_bool_param(pipeline_params, "do_object_segmentation")
    segmentation_model = pipeline_params.string("segmentation_model")
    segmentation_api_port = params.integer("segmentation_api_port")
    segmentation_api_endpoint = f"http://saga03.isi.edu:{segmentation_api_port}"
    segment_colors = parse_bool_param(pipeline_params, "segment_colors")
    refine_colors = parse_bool_param(pipeline_params, "refine_colors")
    strokes_use_refined_colors = parse_bool_param(
        pipeline_params, "strokes_use_refined_colors"
    )
    merge_small_strokes = parse_bool_param(pipeline_params, "merge_small_strokes")
    extract_strokes = parse_bool_param(pipeline_params, "extract_strokes")
    train_gnn = parse_bool_param(pipeline_params, "train_gnn")
    gnn_decode = parse_bool_param(pipeline_params, "gnn_decode")
    email = Email(pipeline_params.string("email")) if "email" in pipeline_params else None
    submission_details_path = pipeline_params.creatable_file("submission_details_path")
    job_logs_path = pipeline_params.creatable_file("job_logs_path")
    if train_gnn:
        model_path = pipeline_params.creatable_file("stroke_model_path")
    elif gnn_decode:
        model_path = pipeline_params.existing_file("stroke_model_path")
    else:
        model_path = None
    base_train_curriculum_path = pipeline_params.existing_directory(
        "base_train_curriculum_path"
    )
    base_test_curriculum_path = pipeline_params.existing_directory(
        "base_test_curriculum_path"
    )
    train_curriculum_path = pipeline_params.creatable_directory("train_curriculum_path")
    test_curriculum_path = pipeline_params.creatable_directory("test_curriculum_path")
    stroke_python_bin_dir = pipeline_params.creatable_directory("stroke_python_bin_dir")
    adam_params_cache_file = pipeline_params.creatable_file("adam_params_cache_file")

    run_id = make_run_identifier(datetime.now().astimezone())
    job_logs_path /= run_id
    job_logs_path.mkdir()

    # We don't also need to swap out run_job_getting_id(). When we run the bash command from this
    # function, it runs synchronously, such that the job is completed once the function returns.
    # So even though it returns None, it doesn't matter because we don't have to keep track of
    # dependencies in that case -- the dependencies are implied by the order the commands are
    # defined.
    command_builder = build_sbatch_command if use_sbatch else build_bash_command
    run_job = run_slurm if use_sbatch else run_bash

    # Truncate the submission details file if it exists, to avoid confusing details with other runs
    # of the same code
    if submission_details_path.exists():
        with submission_details_path.open(mode="w", encoding="utf-8") as _:
            pass
    logger.info(
        "Saving submission details (commands run) to %s.", submission_details_path
    )

    if use_sbatch:
        atexit.register(stop_slurm_jobs)

    if train_curriculum_path == base_train_curriculum_path:
        logger.warning(
            "Output and base train curriculum paths are the same. Overwriting."
        )
    if test_curriculum_path == base_test_curriculum_path:
        logger.warning("Output and base test curriculum paths are the same. Overwriting.")

    splits = ("train", "test")
    split_to_name_to_id = {}
    split_to_base_curriculum_path = {
        "train": base_train_curriculum_path,
        "test": base_test_curriculum_path,
    }
    split_to_curriculum_path = {
        "train": train_curriculum_path,
        "test": test_curriculum_path,
    }

    # Copy base curriculum if needed
    for split, path in split_to_curriculum_path.items():
        if is_empty_dir(path):
            logger.info(
                "Copying split %s from base %s to working copy %s.",
                split,
                split_to_base_curriculum_path[split],
                path,
            )
            # copytree() doesn't like it if the destination already exists, so make sure it doesn't
            path.rmdir()
            copytree(
                split_to_base_curriculum_path[split],
                path,
                ignore=ignore_from_base_curriculum,
            )
        elif not path.is_dir():
            raise RuntimeError(f"Path {path} for split {split} is not a dir.")
        else:
            logger.info("Split %s already exists at %s, not copying.", split, path)

    # Start server
    if do_object_segmentation:
        # We always run the server using Slurm because the segmentation server requires a GPU
        server_job_id = run_slurm(
            echo_command(
                build_sbatch_command(
                    root / "segmentation_processing" / "start_server.sh",
                    script_args=[str(segmentation_api_port)],
                    job_name="adamSegmentServer",
                    log_dir=job_logs_path,
                    email=email,
                    mail_types=[MailType("FAIL")],
                ),
                save_to=submission_details_path,
            ),
            env=os.environ,
            workdir=root / "segmentation_processing",
        )
        minutes_delay_before_segmenting = 2
    # Not really needed, but makes PyCharm happy
    else:
        server_job_id = None
        minutes_delay_before_segmenting = None

    # Run preprocessing
    for split in splits:
        curriculum_path = split_to_curriculum_path[split]
        # PyCharm doesn't understand that we actually reference this key's value later on, such that
        # the dict's complete setup code can't be converted to a dict literal.
        # noinspection PyDictCreation
        name_to_id = {
            "segmentation": run_job(
                echo_command(
                    command_builder(
                        root / "slurm" / "segment.sh",
                        extra_sbatch_args=[
                            f"--dependency=after:{server_job_id}+{minutes_delay_before_segmenting}"
                        ],
                        script_args=[
                            str(curriculum_path),
                            str(curriculum_path),
                            segmentation_api_endpoint,
                            segmentation_model,
                        ],
                        job_name=f"adamSegment{split}",
                        log_dir=job_logs_path,
                        email=email,
                        mail_types=[MailType("FAIL")],
                    ),
                    save_to=submission_details_path,
                ),
                env={"PATH": getenv("PATH", default="")},
                workdir=root,
            )
            if do_object_segmentation
            else None
        }

        name_to_id["color_segmentation"] = (
            run_job(
                echo_command(
                    command_builder(
                        root / "adam_preprocessing" / "color_segment_curriculum.sh",
                        script_args=[
                            str(curriculum_path),
                            str(curriculum_path),
                        ],
                        job_name=f"adamColorSeg{split}",
                        log_dir=job_logs_path,
                        email=email,
                        mail_types=[MailType("FAIL")],
                    ),
                    save_to=submission_details_path,
                ),
                # explicitly add USER to env, because that's what Matlab uses to verify your license
                # -- not whoami etc. -- and subprocess.run() doesn't set USER by default
                env={
                    "PATH": add_to_path(stroke_python_bin_dir),
                    "USER": getenv("USER", default=""),
                },
                workdir=root / "adam_preprocessing",
            )
            if segment_colors
            else None
        )

        name_to_id["color_refinement"] = (
            run_job(
                echo_command(
                    command_builder(
                        root / "adam_preprocessing" / "color_refine_curriculum.sh",
                        script_args=[
                            str(curriculum_path),
                            str(curriculum_path),
                        ],
                        dependencies=dependency_list(
                            name_to_id.get("segmentation"),
                            name_to_id.get("color_segmentation"),
                        ),
                        job_name=f"adamColorRef{split}",
                        log_dir=job_logs_path,
                        email=email,
                        mail_types=[MailType("FAIL")],
                    ),
                    save_to=submission_details_path,
                ),
                # explicitly add USER to env, because that's what Matlab uses to verify your license
                # -- not whoami etc. -- and subprocess.run() doesn't set USER by default
                env={
                    "PATH": add_to_path(stroke_python_bin_dir),
                    "USER": getenv("USER", default=""),
                },
                workdir=root / "adam_preprocessing",
            )
            if refine_colors
            else None
        )

        name_to_id["stroke_extraction"] = (
            run_job(
                echo_command(
                    command_builder(
                        root / "adam_preprocessing" / "extract_strokes.sh",
                        script_args=[
                            str(curriculum_path),
                            str(curriculum_path),
                            "--use-segmentation-type",
                            "color-refined" if strokes_use_refined_colors else "semantic",
                            f"--{'' if merge_small_strokes else 'no-'}merge-small-strokes",
                        ],
                        dependencies=dependency_list(
                            name_to_id.get("segmentation"),
                            name_to_id.get("color_refinement")
                            if strokes_use_refined_colors
                            else None,
                        ),
                        job_name=f"adamStrokes{split}",
                        log_dir=job_logs_path,
                        email=email,
                        mail_types=[MailType("FAIL")],
                    ),
                    save_to=submission_details_path,
                ),
                # explicitly add USER to env, because that's what Matlab uses to verify your license
                # -- not whoami etc. -- and subprocess.run() doesn't set USER by default
                env={
                    "PATH": add_to_path(stroke_python_bin_dir),
                    "USER": getenv("USER", default=""),
                },
                workdir=root / "adam_preprocessing",
            )
            if extract_strokes
            else None
        )
        split_to_name_to_id[split] = name_to_id

    # Stop server after the segmentation jobs are done
    if do_object_segmentation:
        train_segmentation_job = split_to_name_to_id["train"]["segmentation"]
        test_segmentation_job = split_to_name_to_id["test"]["segmentation"]
        run_slurm(
            echo_command(
                # Construct the command directly because it's a weird one-off and doesn't use a script
                [
                    "sbatch",
                    "--account=borrowed",
                    "--partition=scavenge",
                    "--ntasks=1",
                    "--cpus-per-task=1",
                    "--mem-per-cpu=1g",
                    "--requeue",
                    f"--dependency=afterany:{train_segmentation_job}:{test_segmentation_job}",
                    "--job-name=adamCancelServer",
                    f"--output={str(job_logs_path.joinpath('R-%x.%j.out'))}",
                    f"--wrap=scancel {server_job_id}",
                ],
                save_to=submission_details_path,
            ),
            env=os.environ,
            workdir=root,
        )

    # Train GNN
    gnn_train_job = (
        run_job(
            echo_command(
                command_builder(
                    root / "adam_preprocessing" / "train.sh",
                    script_args=[
                        str(split_to_curriculum_path["train"]),
                        str(split_to_curriculum_path["test"]),
                        str(model_path),
                    ],
                    dependencies=dependency_list(
                        split_to_name_to_id["train"].get("stroke_extraction")
                    ),
                    job_name="adamGNNTrain",
                    log_dir=job_logs_path,
                    email=email,
                    mail_types=[MailType("FAIL")],
                ),
                save_to=submission_details_path,
            ),
            env={"PATH": add_to_path(stroke_python_bin_dir)},
            workdir=root / "adam_preprocessing",
        )
        if train_gnn
        else None
    )

    # Get GNN decodes
    for split in splits:
        name_to_id = split_to_name_to_id[split]
        split_curriculum_dir = split_to_curriculum_path[split]
        name_to_id["decode"] = (
            run_job(
                echo_command(
                    command_builder(
                        root / "adam_preprocessing" / "predict.sh",
                        script_args=[
                            str(model_path),
                            str(split_curriculum_dir),
                            str(split_curriculum_dir),
                        ],
                        dependencies=dependency_list(
                            name_to_id.get("stroke_extraction"), gnn_train_job
                        ),
                        job_name=f"adamGNNDecode{split}",
                        log_dir=job_logs_path,
                        email=email,
                        mail_types=[MailType("FAIL")],
                    ),
                    save_to=submission_details_path,
                ),
                env={"PATH": add_to_path(stroke_python_bin_dir)},
                workdir=root / "adam_preprocessing",
            )
            if gnn_decode
            else None
        )

    # Run ADAM train and test
    # Dump the final params file to the given output spot.
    # Note this also includes the pipeline parameters. That's fine -- ADAM shouldn't process them.
    with adam_params_cache_file.open("w", encoding="utf-8") as params_out:
        yaml.dump(params.as_nested_dicts(), params_out)

    run_job(
        echo_command(
            command_builder(
                root / "slurm" / "adam.sh",
                script_args=[str(adam_params_cache_file)],
                dependencies=dependency_list(
                    *[split_to_name_to_id[split].get("decode") for split in splits]
                ),
                job_name="adamADAM",
                log_dir=job_logs_path,
                email=email,
                mail_types=[MailType("SUCCESS"), MailType("FAIL")],
            ),
            save_to=submission_details_path,
        ),
        env=os.environ,
        workdir=root,
    )

    if use_sbatch:
        atexit.unregister(stop_slurm_jobs)


if __name__ == "__main__":
    parameters_only_entry_point(pipeline_entrypoint)
