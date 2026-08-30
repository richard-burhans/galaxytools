"""Open the usegalaxy-tools lockfile bump for a tool release, from this repo's own history.

WHAT IS ALREADY AUTOMATED, AND WHY THIS IS NOT THAT. `usegalaxy-tools` ships the two scripts that do
the actual edit -- `update_tool.py` asks the Tool Shed for the newest installable revision and appends
it, `fix_lockfile.py` sorts and dedupes -- so nobody needs to copy a changeset hash out of a web page.
This orchestrates them; it does not reimplement them, because a second implementation of the lockfile
format is a second thing to keep correct.

WHAT IT ADDS is the part that was still done by hand, and got done wrong in ways worth naming:

1. THE BASE. The previous kegalign bump was SQUASH-merged upstream, so the local branch still read as
   "1 commit ahead" while its content had already landed. Branching from that stale branch stacks an
   already-merged change onto the next PR. This always branches from the upstream default.

2. THE PROVENANCE. A lockfile diff is one line of hex. What a reviewer needs is what that revision
   CONTAINS, and the Tool Shed will not tell you -- its metadata API returned no `valid_tools` for
   either recent kegalign revision. The answer is in THIS repo: the commits touching `tools/<tool>`
   between the previous revision's publish time and the new one's. That correlation is what this
   automates, and it is why the script lives here rather than in the fork.

3. THE CHECK. The diff must be exactly one added line, in the right tool's block. `fix_lockfile.py`
   rewrites the whole file, so "it worked" and "it reformatted 400 unrelated entries" look identical
   until someone reads the diff.

It does NOT open the pull request. It prints the compare URL and stops, because the person who owns
the tool should read the diff before it goes anywhere near a production Galaxy.

    python scripts/release_to_usegalaxy.py --tool kegalign
    python scripts/release_to_usegalaxy.py --tool kegalign --commit --push
"""
import argparse
import json
import pathlib
import re
import shutil
import subprocess
import sys
import urllib.request

import yaml

TOOL_SHED = 'https://toolshed.g2.bx.psu.edu'
DEFAULT_OWNER = 'richard-burhans'
DEFAULT_TOOLSET = 'usegalaxy.org/mapping.yml'


def run(args, cwd, check=True, capture=True):
    """Run a command, returning stdout. Raises on failure unless ``check`` is False."""
    result = subprocess.run(
        args, cwd=str(cwd), check=False, text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )
    if check and result.returncode != 0:
        raise SystemExit(
            'command failed ({0}): {1}\n{2}'.format(
                result.returncode, ' '.join(args), (result.stderr or '').strip()
            )
        )
    return (result.stdout or '').strip()


def shed_revisions(owner, name):
    """``[(numeric, changeset, create_time)]`` for a Tool Shed repository, oldest first.

    Read from the REST API with the standard library rather than through bioblend: this script must
    run before anything is installed, and its own dependency list should not be a reason it cannot.
    """
    url = '{0}/api/repositories?owner={1}&name={2}'.format(TOOL_SHED, owner, name)
    with urllib.request.urlopen(url, timeout=60) as handle:
        matches = json.load(handle)
    repos = [r for r in matches if r.get('name') == name and r.get('owner') == owner]
    if not repos:
        raise SystemExit('no Tool Shed repository {0}/{1}'.format(owner, name))
    url = '{0}/api/repositories/{1}/metadata?downloadable_only=true'.format(TOOL_SHED, repos[0]['id'])
    with urllib.request.urlopen(url, timeout=60) as handle:
        metadata = json.load(handle)
    rows = [
        (v.get('numeric_revision'), v.get('changeset_revision'), v.get('create_time'))
        for v in metadata.values() if isinstance(v, dict)
    ]
    return sorted((r for r in rows if r[0] is not None), key=lambda r: r[0])


def locked_revisions(tools_repo, ref, path, owner, name):
    """The revisions pinned for one tool AT ``ref``, or ``None`` when it lists no such tool.

    Read from the base ref with `git show`, NEVER from the working tree. Reading the checkout makes
    the answer depend on which branch happens to be out: run this while sitting on the branch from a
    previous bump and it reports "already pinned" about a revision upstream has never seen. The
    question is always "does the BASE have it", and the base is a ref, not a directory.
    """
    blob = run(['git', 'show', '{0}:{1}'.format(ref, path)], cwd=tools_repo)
    locked = yaml.safe_load(blob)
    for tool in locked.get('tools', []):
        if tool.get('name') == name and tool.get('owner') == owner:
            return list(tool.get('revisions', []))
    return None


def bioblend_runner(tools_repo):
    """How to invoke the fork's scripts: plain ``python`` if bioblend is importable, else ``uv run``.

    `requirements.txt` pins bioblend to a git commit (bounded retries on HTTP 429). Honour that pin
    rather than pulling whatever PyPI currently ships, since the difference is retry behaviour against
    the very host being queried.
    """
    probe = subprocess.run(
        [sys.executable, '-c', 'import bioblend, yaml'],
        cwd=str(tools_repo), check=False, capture_output=True,
    )
    if probe.returncode == 0:
        return [sys.executable]
    if shutil.which('uv') is None:
        raise SystemExit(
            'bioblend is not importable and uv is not on PATH.\n'
            'Install the fork requirements (pip install -r requirements.txt) and re-run.'
        )
    pin = None
    requirements = tools_repo / 'requirements.txt'
    if requirements.is_file():
        for line in requirements.read_text(encoding='utf-8').splitlines():
            if line.strip().startswith('bioblend'):
                pin = line.strip()
                break
    return ['uv', 'run', '--quiet', '--with', pin or 'bioblend', '--with', 'PyYAML>=4.2', 'python']


def provenance(repo, tool, since, until):
    """Commits in THIS repo touching ``tools/<tool>`` in the window between two published revisions.

    The Tool Shed records when a revision was published, not what went into it. The window between the
    previous publish and this one is the honest answer, and it is a WINDOW rather than a commit: if it
    names more than one commit, the revision may carry all of them, and the caller should say so
    rather than pick the prettiest.
    """
    args = ['git', 'log', '--format=%h %ad %s', '--date=short']
    if since:
        args.append('--since={0}'.format(since))
    if until:
        args.append('--until={0}'.format(until))
    args += ['--', 'tools/{0}'.format(tool)]
    out = run(args, cwd=repo)
    return [line for line in out.splitlines() if line.strip()]


def compare_url(tools_repo, branch, base_branch):
    """The GitHub compare URL that opens a PR from the fork against upstream."""
    def slug(remote):
        url = run(['git', 'remote', 'get-url', remote], cwd=tools_repo, check=False)
        match = re.search(r'[:/]([^/:]+)/([^/]+?)(?:\.git)?$', url or '')
        return (match.group(1), match.group(2)) if match else (None, None)

    up_owner, up_repo = slug('upstream')
    fork_owner, fork_repo = slug('origin')
    if not up_owner or not fork_owner:
        return None
    return (
        'https://github.com/{0}/{1}/compare/{2}...{3}:{4}:{5}?expand=1'
        .format(up_owner, up_repo, base_branch, fork_owner, fork_repo, branch)
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--tool', required=True, help='Tool Shed repository name, e.g. kegalign')
    parser.add_argument('--owner', default=DEFAULT_OWNER, help='Tool Shed owner')
    parser.add_argument('--usegalaxy-tools', default='../usegalaxy-tools',
                        help='path to the usegalaxy-tools checkout (a fork with an `upstream` remote)')
    parser.add_argument('--toolset', default=DEFAULT_TOOLSET, help='toolset yml, relative to that checkout')
    parser.add_argument('--base', default='upstream/master', help='branch to cut from')
    parser.add_argument('--branch', default=None, help='branch name (default: <tool>-<revision>)')
    parser.add_argument('--commit', action='store_true', help='commit the lockfile change')
    parser.add_argument('--push', action='store_true', help='push the branch to origin (implies --commit)')
    args = parser.parse_args()

    here = pathlib.Path(__file__).resolve().parent.parent
    tools_repo = pathlib.Path(args.usegalaxy_tools).expanduser().resolve()
    lockfile = tools_repo / (args.toolset + '.lock')
    if not lockfile.is_file():
        raise SystemExit('no lockfile at {0}'.format(lockfile))
    if run(['git', 'status', '--porcelain'], cwd=tools_repo):
        raise SystemExit('{0} has uncommitted changes; stash them first'.format(tools_repo))

    remote = args.base.split('/')[0]
    base_branch = args.base.split('/', 1)[1] if '/' in args.base else args.base
    run(['git', 'fetch', '--quiet', remote], cwd=tools_repo)

    revisions = shed_revisions(args.owner, args.tool)
    if not revisions:
        raise SystemExit('no installable revisions for {0}/{1}'.format(args.owner, args.tool))
    newest_n, newest, newest_time = revisions[-1]
    previous_time = revisions[-2][2] if len(revisions) > 1 else None
    print('newest installable revision: {0} (numeric {1}, published {2})'
          .format(newest, newest_n, newest_time))

    lock_rel = str(lockfile.relative_to(tools_repo))
    already = locked_revisions(tools_repo, args.base, lock_rel, args.owner, args.tool)
    if already is None:
        raise SystemExit(
            '{0} does not list {1}/{2}. Add it to {3} first -- this script bumps, it does not '
            'introduce a tool.'.format(lockfile.name, args.owner, args.tool, args.toolset)
        )
    if newest in already:
        print('{0} already pins {1}; nothing to do.'.format(args.base, newest))
        return

    branch = args.branch or '{0}-{1}'.format(args.tool, newest)
    run(['git', 'checkout', '-b', branch, args.base], cwd=tools_repo)
    print('branched {0} from {1}'.format(branch, args.base))

    runner = bioblend_runner(tools_repo)
    run(runner + ['scripts/update_tool.py', '--owner', args.owner, '--name', args.tool,
                  args.toolset], cwd=tools_repo, capture=False)
    run(runner + ['scripts/fix_lockfile.py', args.toolset], cwd=tools_repo, capture=False)

    changed = run(['git', 'diff', '--name-only'], cwd=tools_repo).splitlines()
    added = [ln for ln in run(['git', 'diff', '-U0'], cwd=tools_repo).splitlines()
             if ln.startswith('+') and not ln.startswith('+++')]
    expected = lock_rel
    if changed != [expected] or added != ['+  - {0}'.format(newest)]:
        raise SystemExit(
            'REFUSING: expected exactly one added line in {0}.\nfiles: {1}\nadded: {2}\n'
            'Inspect with `git -C {3} diff` and reset the branch.'
            .format(expected, changed, added, tools_repo)
        )
    print('diff verified: one line, {0}'.format(expected))

    window = provenance(here, args.tool, previous_time, newest_time)
    print('\n--- suggested commit message ' + '-' * 44)
    print('Add {0} revision {1} to the toolset\n'.format(args.tool, newest))
    if window:
        print('{0} is the newest installable revision of {1}/{2} on the main ToolShed.'
              .format(newest, args.owner, args.tool))
        print('Commits to tools/{0} in this repo between the previous published revision'
              .format(args.tool))
        print('({0}) and this one:\n'.format(previous_time or 'the beginning'))
        for line in window:
            print('  {0}'.format(line))
        print('\n⚠ VERIFY AND REWRITE. The window above is a correlation by publish time,')
        print('  not a record of what the revision contains. Say what the change DOES.')
    else:
        print('⚠ No commits to tools/{0} in the publish window -- the revision may carry a'
              .format(args.tool))
        print('  macro or dependency change from elsewhere. Establish what it is before merging.')
    print('-' * 74)

    if args.commit or args.push:
        message = 'Add {0} revision {1} to the toolset'.format(args.tool, newest)
        run(['git', 'add', expected], cwd=tools_repo)
        run(['git', 'commit', '-q', '-m', message], cwd=tools_repo)
        print('\ncommitted (placeholder message -- amend it with the real one)')
    if args.push:
        run(['git', 'push', '-u', 'origin', branch], cwd=tools_repo, capture=False)

    url = compare_url(tools_repo, branch, base_branch)
    print('\nopen the PR: {0}'.format(url or '(could not derive; check the `upstream`/`origin` remotes)'))
    if not args.push:
        print('push first: git -C {0} push -u origin {1}'.format(tools_repo, branch))


if __name__ == '__main__':
    main()
