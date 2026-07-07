from moveit_configs_utils import MoveItConfigsBuilder

def walk(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from walk(v, f"{path}.{k}" if path else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from walk(v, f"{path}[{i}]")
    else:
        if isinstance(obj, tuple):
            print(f"⚠️ tuple found at: {path} -> {obj!r}")

if __name__ == "__main__":
    cfg = MoveItConfigsBuilder("meca_500_r3", package_name="meca_with_needle_moveit_config").to_moveit_configs()
    params = cfg.to_dict()
    any_found = False
    for _ in walk(params):
        any_found = True
    if not any_found:
        print("No tuples found in moveit_config dict.")
