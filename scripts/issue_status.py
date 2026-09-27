"""Estado derivado de las issues según sus dependencias (sin etiquetas manuales).

GitHub solo tiene estados OPEN/CLOSED por issue. El 'bloqueo' se deriva:
una issue está LISTA si todas sus dependencias están CLOSED; si no, BLOQUEADA.
Uso: python scripts/issue_status.py
Requiere: gh auth login (usa `gh issue list`).
"""
import json
import subprocess

DEPS = {
    1: [], 2: [],
    3: [1], 4: [1, 2], 5: [1], 6: [1, 2],
    7: [1, 2, 6], 8: [1, 2], 9: [8],
    10: [1, 2], 11: [10, 2],
    12: [5, 6, 7, 8, 9, 10, 11],
    13: [],
    14: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
}


def fetch_states():
    r = subprocess.run(
        ["gh", "issue", "list", "--limit", "50", "--state", "all",
         "--json", "number,state,title"],
        capture_output=True, text=True, check=True)
    return {i["number"]: (i["state"], i["title"]) for i in json.loads(r.stdout)}


def main():
    states = fetch_states()
    print(f"{'#':>3}  {'GitHub':<7} {'Derivado':<9} Título")
    for n in sorted(DEPS):
        gh_state, title = states.get(n, ("?", "?"))
        if gh_state == "CLOSED":
            derived = "HECHA"
        elif all(states.get(d, ("?",))[0] == "CLOSED" for d in DEPS[n]):
            derived = "LISTA"
        else:
            missing = [d for d in DEPS[n] if states.get(d, ("?",))[0] != "CLOSED"]
            derived = f"BLOQUEADA por {','.join(f'#{d}' for d in missing)}"
        print(f"#{n:<3} {gh_state:<7} {derived:<9} {title}")


if __name__ == "__main__":
    main()
