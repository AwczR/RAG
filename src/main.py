import argparse
from src.query import ask
from src.ingest import run_ingest


def _run_query(cfg_path, q, show_ctx=False, json_ctx=False):
    print(ask(cfg_path, q, show_context=show_ctx, as_json=json_ctx))


def main():
    ap = argparse.ArgumentParser("RAG with SiliconFlow")
    ap.add_argument("-c", "--config", required=True, help="path to config.yaml")
    sub = ap.add_subparsers(dest="cmd")

    # ingest
    p1 = sub.add_parser("ingest")
    p1.add_argument("--rebuild", action="store_true")
    p1.set_defaults(func=lambda a: run_ingest(a.config, a.rebuild))

    # ask
    p2 = sub.add_parser("ask")
    p2.add_argument("--q", required=False, help="query string")
    p2.add_argument("q_positional", nargs="?", help="query string (positional)")
    p2.add_argument("--show-context", action="store_true", help="print retrieved contexts")
    p2.add_argument("--json-context", action="store_true", help="contexts as JSON")

    def _ask(a):
        q = a.q or a.q_positional
        if not q:
            raise SystemExit("error: missing query. use --q '...' or positional")
        _run_query(a.config, q, show_ctx=a.show_context, json_ctx=a.json_context)

    p2.set_defaults(func=_ask)

    args = ap.parse_args()
    if not args.cmd:
        ap.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()