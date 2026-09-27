"""Estado de issues según el bloqueo nativo de GitHub (blocked by).

Uso: python scripts/issue_status.py
Requiere: gh auth login.
Lee `issue_dependencies_summary` + `state` vía API: una issue OPEN con
dependencias sin cerrar sale BLOQUEADA; sin dependencias pendientes, LISTA.
"""
import json
import subprocess


def fetch():
    r = subprocess.run(
        ["gh", "api", "repos/santi440/mateIV/issues", "--paginate",
         "--jq", ".[] | {number, state, title, blocked: .issue_dependencies_summary.blocked_by}"],
        capture_output=True, text=True, check=True)
    return [json.loads(line) for line in r.stdout.splitlines() if line.strip()]


def main():
    rows = sorted(fetch(), key=lambda d: d["number"])
    by_num = {d["number"]: d for d in rows}
    print(f"{'#':>3}  {'GitHub':<7} {'Derivado':<9} Título")
    for d in rows:
        n = d["number"]
        if d["state"] == "closed":
            derived = "HECHA"
        elif d["blocked"] == 0:
            derived = "LISTA"
        else:
            derived = f"BLOQUEADA x{d['blocked']}"
        _ = by_num
        print(f"#{n:<3} {d['state']:<7} {derived:<9} {d['title']}")


if __name__ == "__main__":
    main()
