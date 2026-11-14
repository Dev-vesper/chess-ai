import glob


def pgn_files(path):
    if path.endswith(".pgn"):
        return [path]
    return sorted(glob.glob(path.rstrip("/") + "/*.pgn"))
