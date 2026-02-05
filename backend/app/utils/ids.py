import time, uuid
def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
def new_id(prefix: str):
    return f"{prefix}_{uuid.uuid4().hex[:12]}"
