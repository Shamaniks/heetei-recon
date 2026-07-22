import numpy as np
from dataclasses import dataclass, field
from typing import List


@dataclass
class TherionSurveyData:
    survey_name: str = "survey"
    declination: float = 0.0

    centerline_records: List[str] = field(default_factory=list)
    wall_records: List[str] = field(default_factory=list)

    def add_centerline_record(self, s_from: int, s_to: int, length: float, compass: float, clino: float):
        line = f"    {s_from:<5} {s_to:<5} {length:.2f} {compass:.1f} {clino:+.1f}"
        self.centerline_records.append(line)

    def add_wall_shot(self, s_from: int, length: float, compass: float, clino: float):
        line = f"    {s_from:<5} .     {length:.2f} {compass:.1f} {clino:+.1f}"
        self.wall_records.append(line)

    def serialize(self) -> str:
        header = (
            f"survey {self.survey_name}\n"
            "  centerline\n"
            "    units length meters\n"
            "    units compass clino degrees\n"
            f"    declination {self.declination:.2f} degrees\n\n"
            "    data normal from to length compass clino\n"
            "    extend start 0"
        )
        
        centerline_data = "\n".join(self.centerline_records)
        
        walls_header = (
            "    data polar orientation-stats\n"
        )
        
        walls_data = "\n".join(self.wall_records)
        footer = "\n  endcenterline\nendsurvey\n"
        
        return header + centerline_data + walls_header + walls_data + footer

    def save(self):
        file_path = f"{self.survey_name}.th"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(self.serialize())
        print(f"[therion] survey saved: {file_path}")


def cartesian_to_polar(v: np.ndarray) -> np.ndarray:
    lengths = np.sqrt(v[:, 0]**2 + v[:, 1]**2 + v[:, 2]**2)

    valid_mask = lengths > 1e-5
    safe_lengths = np.where(valid_mask, lengths, 1.0)

    compass = np.degrees(np.arctan2(v[:, 0], v[:, 1])) % 360

    clino = np.degrees(np.arcsin(v[:, 2] / safe_lengths))

    compass = np.where(valid_mask, compass, 0.0)
    clino = np.where(valid_mask, clino, 0.0)

    return np.column_stack((lengths, compass, clino))


def compute_step_geometry(p1: np.ndarray, p2: np.ndarray, scale: np.ndarray) -> tuple:
    v = (p2 - p1) * scale
    length = np.linalg.norm(v)
    
    if length < 1e-4:
        return None
        
    compass = np.degrees(np.arctan2(v[0], v[1])) % 360
    clino = np.degrees(np.arcsin(v[2] / length))
    
    return length, compass, clino


def process_cloud_to_therion(
        cloud_xyz: np.ndarray, 
        cloud_gps: np.ndarray, 
        track_xyz: np.ndarray, 
        track_gps: np.ndarray, 
        scale:     np.ndarray,
        survey_name: str,
    ) -> TherionSurveyData:
        
    survey = TherionSurveyData(survey_name=survey_name)

    mid_points = (track_gps[1:] + track_gps[:-1]) / 2

    left_gps_bounds = np.empty_like(track_gps, dtype=track_gps.dtype)
    right_gps_bounds = np.empty_like(track_gps, dtype=track_gps.dtype)
    
    left_gps_bounds[1:] = mid_points
    right_gps_bounds[:-1] = mid_points
    left_gps_bounds[0] = 0
    right_gps_bounds[-1] = np.iinfo(track_gps.dtype).max

    left_boundaries = np.searchsorted(cloud_gps, left_gps_bounds, side='left')
    right_boundaries = np.searchsorted(cloud_gps, right_gps_bounds, side='right')
    
    for j in range(len(track_xyz) - 1):
        p1, p2 = track_xyz[j], track_xyz[j+1]
        
        mag_vector = ((p2 - p1) * scale).reshape(1, 3)
        mag_polar = cartesian_to_polar(mag_vector)[0]
        survey.add_centerline_record(
            s_from=j, 
            s_to=j + 1, 
            length=float(mag_polar[0]), 
            compass=float(mag_polar[1]), 
            clino=float(mag_polar[2])
        )

        start_idx = left_boundaries[j]
        end_idx = right_boundaries[j]
        local_pts = cloud_xyz[start_idx:end_idx]

        if len(local_pts) == 0: continue

        vectors_slice = (local_pts - p1) * scale
        walls_polar = cartesian_to_polar(vectors_slice)

        for shot in walls_polar:
            survey.add_wall_shot(
                s_from=j, 
                length=shot[0], 
                compass=shot[1], 
                clino=shot[2]
            )
        
    return survey
