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

    python scripts/check_usegalaxy_propagation.py --tool kegalign
    python scripts/check_usegalaxy_propagation.py --tool kegalign --watch
"""
import argparse
import json
import sys
import time
import urllib.parse
import urllib.request

TOOL_SHED = "https://toolshed.g2.bx.psu.edu"
DEFAULT_SERVER = "https://usegalaxy.org"
DEFAULT_OWNER = "richard-burhans"
DEFAULT_INTERVAL = 900          # 15 min; CVMFS propagation is measured in hours, not seconds
DEFAULT_TIMEOUT = 12 * 3600


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


def report(server: str, owner: str, name: str) -> bool:
    changeset, want = published_version(owner, name)
    have = installed_versions(server, owner, name)
    host = urllib.parse.urlparse(server).netloc
    print(f"  Tool Shed  {owner}/{name} newest installable: {changeset} "
          f"({want or 'no tool version in its metadata'})")
    print(f"  {host}: {', '.join(have) if have else 'the tool is not installed at all'}")
    if not want:
        print("  UNDECIDABLE — the published revision reports no tool version to compare against")
        return False
    if want in have:
        print(f"  PROPAGATED — {host} is serving {want}")
        return True
    print(f"  NOT YET — {host} has not picked up {want}")
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tool", required=True, help="Tool Shed repository name, e.g. kegalign")
    parser.add_argument("--owner", default=DEFAULT_OWNER)
    parser.add_argument("--server", default=DEFAULT_SERVER,
                        help=f"Galaxy server to check (default {DEFAULT_SERVER})")
    parser.add_argument("--watch", action="store_true",
                        help="keep checking until it propagates")
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL,
                        help=f"seconds between checks with --watch (default {DEFAULT_INTERVAL})")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                        help=f"give up after this many seconds (default {DEFAULT_TIMEOUT})")
    args = parser.parse_args(argv)

    if not args.watch:
        return 0 if report(args.server, args.owner, args.tool) else 1

    started = time.time()
    while True:
        print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}]", flush=True)
        try:
            if report(args.server, args.owner, args.tool):
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
