#!/usr/bin/env python

import argparse
import collections.abc
import concurrent.futures
import contextlib
import io
import json
import multiprocessing
import os
import pathlib
import queue
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import typing

#: Compression level for the concatenated output.
#:
#: ⚠ LEVEL 1, DELIBERATELY -- do not "fix" it back to the zlib default of 6. This single write is
#: THE COLLAPSE: every per-command output file concatenated into one gzip stream, measured at ~62%
#: of this job's wall clock (twice, on inputs differing 9x). Measured on real Cannabis AXT, 1.69 GB
#: through one core: level 6 writes at 18 MB/s, level 1 at 137. Object store is not the constraint;
#: the write is.
#:
#: ⚠ AND IT STAYS AT 1 NOW THAT THE WRITE IS PARALLEL, which is the opposite of what it looks like.
#: With threads, level 6 becomes affordable -- 547 MB/s at 30 threads, four times faster than the
#: SINGLE-threaded level 1 this replaces, and 40% smaller. But level 1 with the same threads is
#: 2,456 MB/s, so against each other level 1 still wins on the axis that matters by 4.5x. Level 6
#: is the right choice only if the bottleneck moves to I/O, where the smaller output would pay for
#: itself -- ⚠ UNMEASURED, because the benchmark above read from page cache.
COMPRESSLEVEL: typing.Final = 1


@contextlib.contextmanager
def open_file(filename: str, threads: int = 1) -> collections.abc.Iterator[typing.IO[str]]:
    """The output handle. Gzipped output goes through `pigz`, which is why `threads` is here.

    ⛔ THIS IS THE COLLAPSE, AND IT WAS SINGLE-THREADED. Python's `gzip` module has no parallel
    mode, so the one write that dominates this job used one core while the other 29 sat idle --
    the compute workers have all finished by the time this runs. Measured on 1.69 GB of real
    Cannabis AXT:

        gzip -6 (the original)      18 MB/s
        gzip -1                    137 MB/s
        pigz -1 -p 30            2,456 MB/s

    ⚠ `-n` IS NOT OPTIONAL. pigz stores the input's name and modification time in the gzip header
    by default, where Python's `gzip.open` did not; without `-n` the output bytes would stop being
    a function of the content alone and two identical runs would differ. Measured, and it is the
    one flag whose absence nothing downstream would complain about.

    ⚠ `-i`/`--independent` IS DELIBERATELY ABSENT. pigz loads the previous block's last 32 KiB as
    a preset dictionary, which is why its output is the same SIZE as gzip's at the same level
    (measured: 0.53 GB, 3.17x, for both). `-i` would drop that for random access and partial error
    recovery, neither of which a single streamed output needs.

    ⚠ pigz reads the `GZIP` and `PIGZ` environment variables BEFORE its command line, so anything
    set there silently overrides these flags. They are cleared for the child rather than trusted.

    ▶ Decompression is NOT helped by any of this -- pigz's own manual says it "can't be
    parallelized" -- so the tarball this job unpacks is unaffected. Compression only.
    """
    if not filename.endswith(".gz"):
        with open(filename, "w") as f:
            yield f
        return

    pigz = shutil.which("pigz")
    if pigz is None:
        # ⛔ LOUD, NOT A FALLBACK. A quiet drop back to Python's gzip would still produce correct
        # output, several times slower, and would hide exactly the failure this wrapper's
        # `pigz` requirement exists to prevent -- a container that resolves without it.
        sys.exit(
            "ERROR: pigz is not on PATH. This tool requires it (see the pigz requirement in "
            "batched_lastz.xml); the environment the wrapper resolved does not provide it."
        )

    env = {k: v for k, v in os.environ.items() if k not in ("GZIP", "PIGZ")}
    with open(filename, "wb") as raw:
        process = subprocess.Popen(
            [pigz, f"-{COMPRESSLEVEL}", "-n", "-p", str(max(1, threads)), "-c"],
            stdin=subprocess.PIPE,
            stdout=raw,
            env=env,
        )
        if process.stdin is None:
            sys.exit("ERROR: could not open a pipe to pigz")
        # ⚠ `newline=""` so nothing rewrites the line endings the callers already wrote; the
        # `gzip.open(..., "wt")` this replaces did not translate them either.
        writer = typing.cast(typing.IO[str], io.TextIOWrapper(process.stdin, newline=""))
        try:
            yield writer
        finally:
            writer.close()
            returncode = process.wait()
        # ⛔ CHECKED. A pigz that died mid-stream leaves a short but perfectly valid .gz behind,
        # and every reader downstream would accept it -- the same shape as the lastz failure mode
        # this project has already been caught by, where a FAILURE still writes partial output.
        if returncode != 0:
            sys.exit(f"ERROR: pigz exited {returncode} while writing {filename}")


lastz_output_format_regex = re.compile(
    r"^(?:axt\+?|blastn|cigar|differences|general-?.+|lav|lav\+text|maf[-+]?|none|paf(?::wfmash)?|rdotplot|sam-?|softsam-?|text)$",
    re.IGNORECASE,
)


# Specifies the output format: lav, lav+text, axt, axt+, maf, maf+, maf-, sam, softsam, sam-, softsam-, cigar, BLASTN, PAF, PAF:wfmash, differences, rdotplot, text, general[:<fields>], or general-[:<fields>].
# --format=none can be used when no alignment output is desired.


def command_succeeded(returncode: int, stderr_file: str | None, stderr_ok: bool) -> bool:
    """Whether one lastz invocation is to be treated as successful.

    ⚠ 1 IS lastz's OWN FAILURE CODE -- suicidef() and chastise() both exit EXIT_FAILURE. It is
    tolerated only because the stderr check can tell a truncation warning from a real error, and
    it can only do that when stderr was captured. With no stderr file there is nothing to
    tolerate it by, so a nonzero code has to be fatal -- otherwise a genuinely failed command is
    indistinguishable from a truncated-but-usable one.

    ⛔ THIS IS A SECOND COPY, AND THE COPY THAT ACTUALLY RUNS. KegAlign's
    `scripts/run_lastz_tarball.py` carries the same function, and
    `tests/test_lastz_failure_propagation.py` imports and exercises THAT one. batched_lastz.xml
    runs THIS file through `$__tool_directory__`, so until now the rule the test guarantees and
    the rule that executed were merely similar-looking inline code. Same name and same shape so a
    change to one is greppable in the other; they still have to be changed together.
    """
    if stderr_file is None:
        return returncode == 0

    return returncode in (0, 1) and stderr_ok


def run_command(
    input_queue: "queue.Queue[dict[str, typing.Any]]",
    output_queue: "queue.Queue[float]",
) -> list[str]:
    os.chdir("galaxy/files")

    # These are not considered errors even though
    # we will end up with a segmented alignment
    truncation_regex = re.compile(
        r"truncating alignment (ending|starting) at \(\d+,\d+\);  anchor at \(\d+,\d+\)$"
    )
    truncation_msg = "truncation can be reduced by using --allocate:traceback to increase traceback memory"

    # ⛔ COLLECT FAILURES, DO NOT RETURN ON THE FIRST ONE. Returning early leaves this
    # worker's sentinel in the queue; another worker then consumes it and exits too, so
    # commands still queued are silently never run. One failed lastz could drop an
    # arbitrary share of the batch.
    failures: list[str] = []

    while True:
        command_dict = input_queue.get()

        if not command_dict:
            return failures

        # 2G, not 1.99G. lastz caps traceback at INT_MAX and carries a special case so
        # that "2G" is accepted and clamped to it. Below 1.04.41 there was no guard and
        # the value was parsed into a signed int, so 2G overflowed -- which is why 1.99G
        # was correct then. The requirement above now pins >= 1.04.52.
        args = ["lastz", "--allocate:traceback=2G"]
        args.extend(command_dict["args"])

        stdin = command_dict["stdin"]
        if stdin is not None:
            stdin = open(stdin)

        stdout = command_dict["stdout"]
        if stdout is not None:
            stdout = open(stdout, "w")

        stderr = command_dict["stderr"]
        if stderr is not None:
            stderr = open(stderr, "w")

        begin = time.perf_counter()
        p = subprocess.run(args, stdin=stdin, stdout=stdout, stderr=stderr)

        for var in [stdin, stdout, stderr]:
            if var is not None:
                var.close()

        # if there is a stderr_file, make sure it is
        # empty or only contains truncation messages
        stderr_ok = True
        stderr_file = command_dict["stderr"]

        if stderr_file is not None:
            try:
                stderr_path = pathlib.Path(stderr_file)
                if stderr_path.lstat().st_size != 0:
                    with stderr_path.open() as f:
                        for stderr_line in f:
                            stderr_line = stderr_line.strip()
                            if not stderr_line:
                                # a blank line is not a diagnostic
                                continue
                            if (not truncation_regex.match(stderr_line) and stderr_line != truncation_msg):
                                stderr_ok = False
            except OSError:
                # cannot read what lastz said, so cannot clear it of being an error
                stderr_ok = False

        if command_succeeded(p.returncode, stderr_file, stderr_ok):
            elapsed = time.perf_counter() - begin
            output_queue.put(elapsed)
        else:
            failures.append(f"command failed (rc={p.returncode}): {' '.join(args)}")


def collect_failures(
    futures: collections.abc.Iterable["concurrent.futures.Future[list[str]]"],
) -> list[str]:
    """Every failure the workers reported, as messages.

    ⛔ A WORKER SIGNALS FAILURE BY RETURNING, NOT BY RAISING. The previous version of this
    logic tested only future STATE -- `done()`, `cancelled()`, `exception()` -- none of
    which is ever true for a worker that returns normally. So a failed lastz was printed to
    stderr and the tool still exited 0, and because the wrapper sets
    `detect_errors="exit_code"` Galaxy marked the job green and handed back a silently
    incomplete alignment.

    Split out so it can be tested without processes, a tarball or lastz.
    """
    failures: list[str] = []

    for future in futures:
        if future.cancelled():
            failures.append("worker was cancelled")
            continue

        exception = future.exception()
        if exception is not None:
            failures.append(f"worker raised: {exception}")
            continue

        failures.extend(future.result())

    return failures


class BatchTar:
    def __init__(self, pathname: str, debug: bool = False) -> None:
        self.pathname = pathname
        self.debug = debug
        self.commands: list[dict[str, typing.Any]] = []
        self.format_name = "tabular"
        self._extract()
        self._load_commands()
        self._load_format()

    def batch_commands(self) -> collections.abc.Iterator[dict[str, typing.Any]]:
        yield from self.commands

    def final_output_format(self) -> str:
        return self.format_name

    def _extract(self) -> None:
        try:
            self.tarball = tarfile.open(
                name=self.pathname, mode="r:*", format=tarfile.GNU_FORMAT
            )
        except FileNotFoundError:
            sys.exit(f"ERROR: unable to find input tarball: {self.pathname}")
        except tarfile.ReadError:
            sys.exit(f"ERROR: error reading input tarball: {self.pathname}")

        begin = time.perf_counter()
        self.tarball.extractall(filter="data")
        self.tarball.close()
        elapsed = time.perf_counter() - begin

        if self.debug:
            print(
                f"Extracted tarball in {elapsed} seconds", file=sys.stderr, flush=True
            )

    def _load_commands(self) -> None:
        commands_path = pathlib.Path("galaxy/commands.json")
        if not commands_path.is_file():
            sys.exit(
                f"ERROR: input tarball missing galaxy/commands.json: {self.pathname}"
            )

        begin = time.perf_counter()
        f = commands_path.open()
        for json_line in f:
            json_line = json_line.rstrip("\n")
            try:
                command_dict = json.loads(json_line)
            except json.JSONDecodeError:
                sys.exit(
                    f"ERROR: bad json line in galaxy/commands.json: {self.pathname}"
                )

            self._load_command(command_dict)

        f.close()
        elapsed = time.perf_counter() - begin

        if self.debug:
            print(
                f"loaded {len(self.commands)} commands in {elapsed} seconds ",
                file=sys.stderr,
                flush=True,
            )

    def _load_command(self, command_dict: dict[str, typing.Any]) -> None:
        # check command_dict structure
        field_types: dict[str, list[typing.Any]] = {
            "executable": [str],
            "args": [list],
            "stdin": [str, "None"],
            "stdout": [str, "None"],
            "stderr": [str, "None"],
        }

        bad_format = False
        for field_name in field_types.keys():
            # missing field
            if field_name not in command_dict:
                bad_format = True
                break

            # incorrect field type
            good_type = False
            for field_type in field_types[field_name]:
                if isinstance(field_type, str) and field_type == "None":
                    if command_dict[field_name] is None:
                        good_type = True
                        break
                elif isinstance(command_dict[field_name], field_type):
                    good_type = True
                    break

            if good_type is False:
                bad_format = True

        if not bad_format:
            # all args must be strings
            for arg in command_dict["args"]:
                if not isinstance(arg, str):
                    bad_format = True
                    break

        if bad_format:
            sys.exit(
                f"ERROR: unexpected json format in line in galaxy/commands.json: {self.pathname}"
            )

        self.commands.append(command_dict)

    def _load_format(self) -> None:
        try:
            with open("galaxy/format.txt") as f:
                format_name = f.readline()
                format_name = format_name.rstrip("\n")
        except FileNotFoundError:
            sys.exit(f"ERROR: input tarball missing galaxy/format.txt: {self.pathname}")

        format_map = {
            "axt": "axt",
            "axt+": "axt",
            "cigar": "cigar",
            "differences": "interval",
            "lav": "lav",
            "lav+text": "lav",
            "maf": "maf",
            "maf+": "maf",
            "maf-": "maf",
            "sam": "sam",
            "sam-": "sam",
            "softsam": "sam",
            "softsam-": "sam"
        }

        self.format_name = format_map.get(format_name, "tabular")


class TarRunner:
    def __init__(
        self,
        input_pathname: str,
        output_pathname: str,
        parallel: int,
        debug: bool = False,
    ) -> None:
        self.input_pathname = input_pathname
        self.output_pathname = output_pathname
        self.parallel = parallel
        self.debug = debug
        self.batch_tar = BatchTar(self.input_pathname, debug=self.debug)
        self.output_file_format: dict[str, str] = {}
        self.output_files: dict[str, list[str]] = {}
        self._set_output()
        self._set_target_query()

    def _set_output(self) -> None:
        for command_dict in self.batch_tar.batch_commands():
            output_file = None
            output_format = None

            for arg in command_dict["args"]:
                if arg.startswith("--format="):
                    output_format = arg[9:]
                elif arg.startswith("--output="):
                    output_file = arg[9:]

            if output_file is None:
                f = tempfile.NamedTemporaryFile(dir="galaxy/files", delete=False)
                output_file = pathlib.Path(f.name).name
                f.close()
                command_dict["args"].append(f"--output={output_file}")

            if output_format is None:
                output_format = "lav"
                command_dict["args"].append(f"--format={output_format}")

            if not lastz_output_format_regex.match(output_format):
                sys.exit(f"ERROR: invalid output format: {output_format}")

            self.output_file_format[output_file] = output_format

        for output_file, output_format in self.output_file_format.items():
            self.output_files.setdefault(output_format, [])
            self.output_files[output_format].append(output_file)

    def _set_target_query(self) -> None:
        for command_dict in self.batch_tar.batch_commands():
            new_args: list[str] = []

            for arg in command_dict["args"]:
                if arg.startswith("--target="):
                    new_args.insert(0, arg[9:])
                elif arg.startswith("--query="):
                    new_args.insert(1, arg[8:])
                else:
                    new_args.append(arg)

            command_dict["args"] = new_args

    def run(self) -> None:
        run_times = []
        begin = time.perf_counter()

        with multiprocessing.Manager() as manager:
            input_queue: queue.Queue[dict[str, typing.Any]] = manager.Queue()
            output_queue: queue.Queue[float] = manager.Queue()

            for command_dict in self.batch_tar.batch_commands():
                input_queue.put(command_dict)

            # use the empty dict as a sentinel
            for _ in range(self.parallel):
                input_queue.put({})

            with concurrent.futures.ProcessPoolExecutor(
                max_workers=self.parallel
            ) as executor:
                futures = [
                    executor.submit(run_command, input_queue, output_queue)
                    for _ in range(self.parallel)
                ]

            failures = collect_failures(concurrent.futures.as_completed(futures))
            for failure in failures:
                print(f"lastz: {failure}", file=sys.stderr, flush=True)
            found_failures = bool(failures)

            while not output_queue.empty():
                run_time = output_queue.get()
                run_times.append(run_time)
                if self.debug:
                    print(f"lastz took {run_time}", file=sys.stderr, flush=True)

            if found_failures:
                sys.exit("lastz command failed")

        elapsed = time.perf_counter() - begin

        if self.debug:
            print(f"elapsed {elapsed}", file=sys.stderr, flush=True)

        self._cleanup()

    def _cleanup(self) -> None:
        num_output_files = len(self.output_files.keys())
        if num_output_files != 1:
            sys.exit(f"ERROR: expecting a single output file, found {num_output_files}")

        final_output_format = self.batch_tar.final_output_format()
        if final_output_format in ["axt", "maf"]:
            final_output_format = f"{final_output_format}.gz"

        for file_type, file_list in self.output_files.items():
            # ▶ The compute workers are all finished by now, so the whole slot budget is free
            # for the one write that is left. That is the entire point of the change.
            with open_file(f"output.{final_output_format}", threads=self.parallel) as ofh:
                if final_output_format == "maf.gz":
                    print("##maf version=1", file=ofh)

                for filename in file_list:
                    with open(f"galaxy/files/{filename}") as ifh:
                        for line in ifh:
                            ofh.write(line)

        # move rather than copy: this file can be many gigabytes, and copy2 holds two
        # of them on disk at once
        src_filename = f"output.{final_output_format}"
        shutil.move(src_filename, self.output_pathname)

        output_metadata = {
            "output": {
                "ext": final_output_format,
            }
        }

        with open("galaxy.json", "w") as ofh:
            json.dump(output_metadata, ofh)


def main() -> None:
    if not hasattr(tarfile, "data_filter"):
        sys.exit("ERROR: extracting may be unsafe; consider updating Python")

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--parallel", type=int, default=2, required=False)
    parser.add_argument("--debug", action="store_true", required=False)

    args = parser.parse_args()
    runner = TarRunner(args.input, args.output, args.parallel, args.debug)
    runner.run()


if __name__ == "__main__":
    main()
