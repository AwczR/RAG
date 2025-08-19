import yaml, os

def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    rt = cfg.get("runtime", {}) or {}
    base = rt.get("base_url", "https://api.siliconflow.cn/v1")
    key  = rt.get("api_key")
    timeout = rt.get("timeout_sec", 60)

    if not key:
        raise RuntimeError("Missing runtime.api_key in config.yaml")

    os.environ["OPENAI_BASE_URL"] = base
    os.environ["OPENAI_API_KEY"]  = key

    print(f"[CONFIG] base_url={base}")
    print(f"[CONFIG] key head/tail={key[:6]}...{key[-4:]}")

    rt["timeout_sec"] = timeout
    cfg["runtime"] = rt
    return cfg