"""Release a quarantined GPU lease only after its exact Compose owner is stopped.

Runs on the Docker host. This command never guesses a PID or clears all leases.
Inspect first without --apply. With --apply, the named owner must already be
stopped; restart it only after this command succeeds. No passwords are printed.
"""
import argparse
import json
import subprocess
import time


def docker(*args):
    return subprocess.check_output(["docker", *args], text=True, encoding="utf-8", errors="replace", timeout=30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True)
    parser.add_argument("--api-container", required=True)
    parser.add_argument("--owner-container", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if not args.project.startswith("bysj"):
        raise RuntimeError("only a named bysj Compose project is accepted")
    api, owner = (json.loads(docker("inspect", name))[0] for name in (args.api_container, args.owner_container))
    for record in (api, owner):
        if record["Config"].get("Labels", {}).get("com.docker.compose.project") != args.project:
            raise RuntimeError("container is not in the explicitly selected project")
    probe = "from app.db.session import get_session_factory; from app.db.models import ResourceLease; import json; db=get_session_factory()(); r=db.get(ResourceLease,'gpu:0'); print(json.dumps({'holder':r.holder,'generation':r.generation,'job_id':r.job_id,'expires_at':r.expires_at.isoformat()} if r else {}))"
    lease = json.loads(docker("exec", args.api_container, "python", "-c", probe))
    hostname = owner["Config"]["Hostname"][:16]
    if not lease.get("holder") or f"-{hostname}-" not in lease["holder"]:
        raise RuntimeError("lease holder does not match the exact owner container")
    proof = {"project": args.project, "owner_container_id": owner["Id"], "owner_hostname": hostname,
             "owner_running": owner["State"]["Running"], "owner_pid": owner["State"]["Pid"], "lease": lease}
    if not args.apply:
        print(json.dumps({**proof, "action": "inspection_only"}, indent=2)); return
    if owner["State"]["Running"] or owner["State"]["Pid"] != 0:
        raise RuntimeError("stop the exact owner container and confirm its processes exited before recovery")
    # Recheck Docker state immediately before CAS. Owner heartbeat must also have
    # expired; an active worker would invalidate the process-termination proof.
    release = """import json,sys
from app.db.session import get_session_factory
from app.db.models import ResourceLease,JobTask
from app.db.base import utcnow
from app.domains import lease_service,job_service
expected=json.loads(sys.argv[1])
with get_session_factory()() as db:
    active=db.query(ResourceLease).filter(ResourceLease.resource.startswith('worker:'), ResourceLease.holder==expected['holder'].split(':',1)[0], ResourceLease.expires_at>utcnow()).count()
    if active: raise RuntimeError('owner heartbeat still active; wait for expiry after stopping')
    row=db.get(ResourceLease,'gpu:0')
    if not row or (row.holder,row.generation,row.job_id)!=(expected['holder'],expected['generation'],expected['job_id']): raise RuntimeError('lease changed; refuse stale recovery')
    job_service.mark_stale_interrupted(db,older_than_seconds=0,job_id=expected['job_id'])
    changed=lease_service.release_lease(db,'gpu:0',holder=expected['holder'],generation=expected['generation'],job_id=expected['job_id'])
    if not changed: raise RuntimeError('lease changed during recovery')
    job=db.get(JobTask,expected['job_id'])
    if job: job_service.append_event(db,job,'gpu_recovered','Stopped owner container verified; exact fenced lease released',{'generation':expected['generation']})
    print(json.dumps({'released':True,'generation':expected['generation']}))
"""
    latest=json.loads(docker("inspect", args.owner_container))[0]
    if latest["Id"] != owner["Id"] or latest["State"]["Running"] or latest["State"]["Pid"] != 0:
        raise RuntimeError("owner changed or restarted; refuse recovery")
    result=json.loads(docker("exec", args.api_container, "python", "-c", release, json.dumps(lease)))
    print(json.dumps({**proof, **result}, indent=2))


if __name__ == "__main__":
    main()
