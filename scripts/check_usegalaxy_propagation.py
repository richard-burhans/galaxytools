"""Has a Tool Shed revision actually reached usegalaxy.org yet?

WHERE THIS SITS. `release_to_usegalaxy.py` ends at the usegalaxy-tools compare URL. After that
PR merges, nothing has happened on the server yet: the tool reaches usegalaxy.org through CVMFS,
and that propagation is neither instant nor observable from the lockfile. Until it lands, the
production instance is still serving the PREVIOUS version -- which is the state that makes a
release look finished when it is not.

WHAT IT COMPARES, AND WHY BOTH SIDES ARE FETCHED. The published side comes from the Tool Shed's
ordered installable revisions; the installed side from usegalaxy.org's own tool list. Neither is
read from a file in this repo, because a lockfile entry proves only that somebody asked for a
revision, not that the server has it.

⚠ NO CREDENTIALS. `/api/tools` answers anonymously for public tools, so this script carries no key
and can run anywhere. Do not add one: a propagation check is not a reason to put an API key into a
public repository.

⚠ THE VERSION, NOT THE CHANGESET. usegalaxy.org reports a tool's version (`0.3.1+galaxy0`), never
the Tool Shed changeset it came from, so the comparison is on version strings. Two revisions that
ship the SAME tool version are indistinguishable here -- which is a real limit, not an oversight:
a wrapper-only change that does not bump @VERSION_SUFFIX@ is invisible to any observer outside the
server, and that is a reason to bump the suffix, not a reason to trust a different check.

⛔ INSTALLED IS NOT RUNNABLE, AND THIS SCRIPT USED TO CONFLATE THEM. On 2026-09-02 it reported
kegalign 0.3.1+galaxy0 as propagated to usegalaxy.org, correctly -- and every job using it died,
because the mulled Singularity image `kegalign-full:0.3.1--hdfd78af_0` had not been built yet.
Galaxy had the tool and not its container, so jobs fell back to another image and produced a bare
`gzip: invalid magic` with no aligner output at all. Two hours of a validation run went into a
question this check could have answered in one request. So it now checks the image too, in the two
places it has to exist:

  * `depot.galaxyproject.org/singularity/<pkg>:<version>--<build>` -- where the image is BUILT
  * the `singularity.galaxyproject.org` CVMFS repository -- what a job actually MOUNTS, and which
    lags the depot. Compared by publication TIMESTAMP: a CVMFS revision published before the image
    was built cannot contain it.

⚠ The requirement package is not the tool name. The `kegalign` tool requires `kegalign-full`, so
the image to look for is named after the REQUIREMENT. `--package` says which; it defaults to the
tool name, which is right for tools whose names match and wrong silently otherwise.

    python scripts/check_usegalaxy_propagation.py --tool kegalign --package kegalign-full
    python scripts/check_usegalaxy_propagation.py --tool kegalign --watch
"""
import argparse
import datetime
import email.utils
import json
import re
import sys
import time
import urllib.parse
import urllib.request

TOOL_SHED = "https://toolshed.g2.bx.psu.edu"
DEFAULT_SERVER = "https://usegalaxy.org"
DEFAULT_OWNER = "richard-burhans"
DEFAULT_INTERVAL = 900          # 15 min; CVMFS propagation is measured in hours, not seconds
DEFAULT_TIMEOUT = 12 * 3600
DEPOT = "https://depot.galaxyproject.org/singularity"
CVMFS_PUBLISHED = ("http://cvmfs1-psu0.galaxyproject.org/cvmfs/"
                   "singularity.galaxyproject.org/.cvmfspublished")


def get_json(url: str, timeout: int = 60):
    with urllib.request.urlopen(urllib.request.Request(url), timeout=timeout) as response:
        return json.load(response)


def published_version(owner: str, name: str) -> tuple[str, str]:
    """`(changeset, tool_version)` for the newest INSTALLABLE revision on the Tool Shed.

    Installable rather than merely present: `get_ordered_installable_revisions` is the same list
    `usegalaxy-tools`' own `update_tool.py` reads, so this asks the question the lockfile asks.
    """
    query = urllib.parse.urlencode({"name": name, "owner": owner})
    revisions = get_json(f"{TOOL_SHED}/api/repositories/get_ordered_installable_revisions?{query}")
    if not revisions:
        raise SystemExit(f"no installable revisions for {owner}/{name} on the Tool Shed")
    changeset = revisions[-1]

    repos = get_json(f"{TOOL_SHED}/api/repositories?{query}")
    repos = repos if isinstance(repos, list) else [repos]
    if not repos:
        raise SystemExit(f"Tool Shed has no repository {owner}/{name}")
    metadata = get_json(f"{TOOL_SHED}/api/repositories/{repos[0]['id']}/metadata")
    for entry in metadata.values():
        if entry.get("changeset_revision") == changeset:
            versions = sorted({t.get("version") for t in entry.get("tools") or [] if t.get("version")})
            if versions:
                return changeset, versions[-1]
    # A revision with no tool version is a real answer -- report it rather than inventing one.
    return changeset, ""


def installed_versions(server: str, owner: str, name: str) -> list[str]:
    """Every version of the tool the server currently offers, newest last."""
    tools = get_json(f"{server.rstrip('/')}/api/tools?{urllib.parse.urlencode({'q': name})}")
    ids = {(t if isinstance(t, str) else t.get("id")) or "" for t in tools}
    prefix = f"repos/{owner}/{name}/"
    return sorted({i.rsplit("/", 1)[-1] for i in ids if prefix in i})


def container_status(package: str, version: str) -> tuple[bool, str]:
    """Is the mulled image for `package` `version` built, and has CVMFS caught up?

    Returns (usable, explanation). "Usable" means the image exists on the depot AND the CVMFS
    repository has published a revision since it was built -- a job mounts CVMFS, not the depot,
    so a built-but-unpublished image is exactly as unrunnable as a missing one.

    The build string is not knowable from the version alone, so the depot's directory listing is
    searched for any tag matching the version. That is one request and it is exact; guessing a
    build string and getting a 404 would report "not built" for an image that is.
    """
    prefix = f"{package}:{version}--"
    try:
        listing = urllib.request.urlopen(
            urllib.request.Request(DEPOT + "/"), timeout=90).read().decode("utf-8", "replace")
    except OSError as e:
        return False, f"could not read the depot listing ({type(e).__name__}: {e})"
    tags = sorted({m.group(0) for m in re.finditer(
        re.escape(prefix) + r"[A-Za-z0-9_]+", listing)})
    if not tags:
        return False, f"no image {prefix}* has been built yet"

    try:
        with urllib.request.urlopen(urllib.request.Request(DEPOT + "/" + tags[-1]),
                                    timeout=90) as response:
            built_header = response.headers.get("Last-Modified")
    except OSError as e:
        return False, f"{tags[-1]} is listed but not fetchable ({type(e).__name__}: {e})"
    built = (email.utils.parsedate_to_datetime(built_header).timestamp()
             if built_header else None)

    try:
        published = urllib.request.urlopen(
            urllib.request.Request(CVMFS_PUBLISHED), timeout=90).read().decode("utf-8", "replace")
    except OSError as e:
        return False, f"{tags[-1]} is built, but CVMFS is unreachable ({type(e).__name__}: {e})"
    revision = re.search(r"^S(\d+)", published, re.M)
    stamp = re.search(r"^T(\d+)", published, re.M)
    if not stamp:
        return False, f"{tags[-1]} is built, but CVMFS published no timestamp to compare against"
    cvmfs_time = int(stamp.group(1))
    rev = revision.group(1) if revision else "?"
    if built is None:
        return False, f"{tags[-1]} is built, but the depot served no Last-Modified to compare"
    if cvmfs_time < built:
        return False, (f"{tags[-1]} was built {_utc(built)} but CVMFS is still at S{rev} "
                       f"({_utc(cvmfs_time)}) -- a job would NOT find the image")
    return True, f"{tags[-1]} built {_utc(built)}, CVMFS S{rev} published {_utc(cvmfs_time)}"


def _utc(epoch: float) -> str:
    return datetime.datetime.fromtimestamp(epoch, datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def report(server: str, owner: str, name: str, package: str) -> bool:
    changeset, want = published_version(owner, name)
    have = installed_versions(server, owner, name)
    host = urllib.parse.urlparse(server).netloc
    print(f"  Tool Shed  {owner}/{name} newest installable: {changeset} "
          f"({want or 'no tool version in its metadata'})")
    print(f"  {host}: {', '.join(have) if have else 'the tool is not installed at all'}")
    if not want:
        print("  UNDECIDABLE — the published revision reports no tool version to compare against")
        return False
    if want not in have:
        print(f"  NOT YET — {host} has not picked up {want}")
        return False

    # ⛔ The tool being installed is where this check used to stop, and it was not enough.
    upstream = want.split("+")[0]
    usable, detail = container_status(package, upstream)
    print(f"  container  {package} {upstream}: {detail}")
    if not usable:
        print(f"  INSTALLED BUT NOT RUNNABLE — {host} serves {want}, but a job cannot mount its "
              f"image. This is the state that produces a bare error with no tool output.")
        return False
    print(f"  PROPAGATED — {host} is serving {want} and its container is mountable")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tool", required=True, help="Tool Shed repository name, e.g. kegalign")
    parser.add_argument("--owner", default=DEFAULT_OWNER)
    parser.add_argument("--package", default=None,
                        help="conda package whose mulled image backs the tool (default: the tool "
                             "name; kegalign is backed by kegalign-full)")
    parser.add_argument("--server", default=DEFAULT_SERVER,
                        help=f"Galaxy server to check (default {DEFAULT_SERVER})")
    parser.add_argument("--watch", action="store_true",
                        help="keep checking until it propagates")
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL,
                        help=f"seconds between checks with --watch (default {DEFAULT_INTERVAL})")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                        help=f"give up after this many seconds (default {DEFAULT_TIMEOUT})")
    args = parser.parse_args(argv)
    package = args.package or args.tool

    if not args.watch:
        return 0 if report(args.server, args.owner, args.tool, package) else 1

    started = time.time()
    while True:
        print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}]", flush=True)
        try:
            if report(args.server, args.owner, args.tool, package):
                return 0
        except (OSError, ValueError) as e:
            # A transient failure is not an answer. Keep waiting rather than reporting
            # "not propagated", which is what a bare except would have made this say.
            print(f"  check failed ({type(e).__name__}: {e}); will retry", flush=True)
        if time.time() - started + args.interval > args.timeout:
            print(f"  giving up after {int((time.time() - started) // 60)} min")
            return 1
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
