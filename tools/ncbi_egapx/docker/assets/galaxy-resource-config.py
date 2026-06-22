#!/usr/bin/env python3

"""Generate a Galaxy resource-limit overlay for the EGAPx Nextflow config.

EGAPx ships an authoritative ``process_resources.config`` that defines the
process tiers/labels (``single_cpu``, ``multi_cpu``, ``multi_node``,
``gpx_submitter``, ``long_job``, ``small_mem``, ``med_mem``, ``large_mem`` ...)
including the ``ext { ... }`` blocks.  Rather than re-emit that whole structure
here (which drifts whenever upstream changes it), this script:

  1. ``includeConfig`` the upstream ``process_resources.config`` verbatim, then
  2. overrides ONLY the values that depend on the Galaxy-allocated resources:
     the ``params``, the default ``process`` cpus/memory, and the memory tiers.

The overlay is appended to ``egapx_config/galaxy.config`` (which already sets
``docker.enabled = false`` and the ``env { ... }`` PATH block at build time),
and EGAPx is then run with ``-e galaxy`` so it picks the file up.
"""

import os
import shutil
import sys
import typing

DEFAULT_CPUS: typing.Final = 32
MINIMUM_CPUS: typing.Final = 4

DEFAULT_MEMORY: typing.Final = 256
MINIMUM_MEMORY: typing.Final = 6

EGAPX_ROOT: typing.Final = "/galaxy/egapx"
CONFIG_DIR: typing.Final = "egapx_config"
# Upstream resource-tier definitions. Included verbatim so the label/ext
# structure stays authoritative upstream; we only override numbers below.
PROCESS_RESOURCES: typing.Final = "process_resources.config"
UPSTREAM_PROCESS_RESOURCES: typing.Final = os.path.join(
    EGAPX_ROOT, "ui", "assets", "config", "user", PROCESS_RESOURCES
)


def get_allocated_cpus(value: str | None) -> int:
    available_cpus = None
    requested_cpus = None

    try:
        available_cpus = len(os.sched_getaffinity(0))
    except Exception:
        pass

    if value is not None:
        try:
            requested_cpus = int(value)
        except Exception:
            pass

    if requested_cpus is None:
        requested_cpus = DEFAULT_CPUS

    if available_cpus is None:
        return requested_cpus
    else:
        return min(requested_cpus, available_cpus)


def get_allocated_memory(value: str | None) -> int:
    available_memory = None
    requested_memory = None

    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemFree:"):
                    _, available_kb, _ = line.split()
                    available_memory = int(available_kb) // 1048576
    except Exception:
        pass

    if value is not None:
        try:
            requested_memory = int(value)
        except Exception:
            pass

    if requested_memory is None:
        requested_memory = DEFAULT_MEMORY

    if available_memory is None:
        return requested_memory
    else:
        return min(requested_memory, available_memory)


# copy default config files into the working directory
shutil.copytree(os.path.join(EGAPX_ROOT, CONFIG_DIR), CONFIG_DIR, dirs_exist_ok=True)

# Make sure the upstream resource-tier config is present so galaxy.config can
# include it (it is normally copied in with the rest of egapx_config, but stage
# it from the source tree as a fallback so the include never dangles).
process_resources_path = os.path.join(CONFIG_DIR, PROCESS_RESOURCES)
if not os.path.exists(process_resources_path) and os.path.exists(UPSTREAM_PROCESS_RESOURCES):
    shutil.copy2(UPSTREAM_PROCESS_RESOURCES, process_resources_path)

# get galaxy env vars
galaxy_slots_env_var = os.environ.get("GALAXY_SLOTS")
galaxy_memory_mb_env_var = os.environ.get("GALAXY_MEMORY_MB")

# gigabase_pairs (1-8) is a pass-through scheduling hint from the tool form. It
# has no built-in resource effect yet; it is simply exposed as a Nextflow param
# for downstream / future use.
try:
    gigabase_pairs = int(os.environ.get("EGAPX_GIGABASE_PAIRS", "1"))
except (TypeError, ValueError):
    gigabase_pairs = 1
gigabase_pairs = min(8, max(1, gigabase_pairs))

galaxy_config_path = os.path.join(CONFIG_DIR, "galaxy.config")

# galaxy is not trying to limit resources: still pass the hint through.
if galaxy_slots_env_var is None and galaxy_memory_mb_env_var is None:
    with open(galaxy_config_path, "a") as of:
        of.write(f"\nparams.gigabase_pairs = {gigabase_pairs}\n")
    sys.exit(0)

galaxy_slots = max(get_allocated_cpus(galaxy_slots_env_var), MINIMUM_CPUS)
galaxy_memory_gb = max(get_allocated_memory(galaxy_memory_mb_env_var), MINIMUM_MEMORY)

galaxy_threads = 16
galaxy_nodes = 16

# Environment-dependent numbers. Floors keep the values valid (Nextflow rejects
# cpus = 0 / memory = 0.GB) on small allocations.
num_cpus_per_node = max(1, galaxy_slots // galaxy_threads)

large_cpu_job = (galaxy_slots - 1) // 2
std_cpu_job = max(1, large_cpu_job // 4)

large_memory_job = max(1, (galaxy_memory_gb - 4) // 2)
med_memory_job = max(1, large_memory_job // 2)
std_memory_job = max(1, large_memory_job // 4)
small_memory_job = 8  # upstream 'small_mem' tier, kept fixed

# Overlay: include the upstream tiers, then override only what depends on the
# Galaxy allocation. The label/ext blocks (single_cpu, multi_cpu, multi_node,
# gpx_submitter, long_job) are inherited from process_resources.config; the
# multi_* labels track params.threads, which we set here.
galaxy_config = f"""
// --- Galaxy resource limits (generated by galaxy-resource-config.py) ---
// Upstream process_resources.config stays authoritative for the process
// tiers/labels; only the environment-dependent numbers are overridden here.
includeConfig "./{PROCESS_RESOURCES}"

params.threads = {galaxy_threads}
params.nodes = {galaxy_nodes}
params.num_cpus_per_node = {num_cpus_per_node}
params.gigabase_pairs = {gigabase_pairs}

process {{
    cpus = {std_cpu_job}
    memory = {std_memory_job}.GB

    withLabel: 'small_mem' {{
        memory = {small_memory_job}.GB
    }}
    withLabel: 'med_mem' {{
        memory = {med_memory_job}.GB
    }}
    withLabel: 'large_mem' {{
        memory = {large_memory_job}.GB
    }}

    // Cap concurrent multi-cpu/multi-node jobs to how many params.threads-sized
    // jobs fit on the node (= num_cpus_per_node). The old script divided
    // num_cpus_per_node by params.threads again, yielding a fractional maxForks.
    withLabel: 'multi_cpu' {{
        maxForks = params.num_cpus_per_node
    }}
    withLabel: 'multi_node' {{
        maxForks = params.num_cpus_per_node
    }}
}}
"""

with open(galaxy_config_path, "a") as of:
    of.write(galaxy_config)
