import json
import os, glob


def get_buffer_size(status_file):
    with open(status_file, "r") as f:
        data = json.load(f)
        points_in_chunk = data["lastLazStatus"]["points_count"]
    return points_in_chunk
