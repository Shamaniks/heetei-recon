import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict
import collections
from scipy.spatial import KDTree


@dataclass
class TherionSurveyData:
    survey_name: str = "survey"
    declination: float = 0.0

    centerline_records: List[str] = field(default_factory=list)
    wall_records: Dict[int, List[str]] = field(
        default_factory=lambda: collections.defaultdict(list)
    )

    def add_centerline_record(self, s_from: int, s_to: int, length: float, compass: float, clino: float):
        line = f"    {s_from:<5} {s_to:<5} {length:.2f} {compass:.1f} {clino:+.1f}"
        self.centerline_records.append(line)


    def add_wall_shot(self, s_from: int, length: float, compass: float, clino: float):
        line = f"    {s_from:<5} .     {length:.2f} {compass:.1f} {clino:+.1f}"
        self.wall_records[s_from].append(line)


    def serialize(self) -> str:
        safe_survey_name = self.survey_name.replace('.', '_')

        header = (
            f"survey {safe_survey_name}\n"
            "  centerline\n"
            "    units length meters\n"
            "    units compass clino degrees\n"
            f"    declination {self.declination:.2f} degrees\n"
            "    extend start 0\n\n"
            "    data normal from to length compass clino\n"
        )
        
        centerline_data = "\n".join(self.centerline_records)
        footer_mag = "\n  endcenterline\n"
        
        walls_blocks = []
        for station_id, shots in self.wall_records.items():
            if not shots: 
                continue
            block = (
                f"\n  centerline\n"
                f"    units length meters\n"
                f"    units compass clino degrees\n"
                f"    data normal from to length compass clino\n" 
                + "\n".join(shots) +
                f"\n  endcenterline"
            )
            walls_blocks.append(block)
            
        return header + centerline_data + footer_mag + "".join(walls_blocks) + "\nendsurvey\n"


    def save(self):
        file_path = f"therion_files/{self.survey_name}.th"
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
    ratio = np.clip(v[:, 2] / safe_lengths, -1.0, 1.0)
    clino = np.degrees(np.arcsin(ratio))
    
    return length, compass, clino


def sort_walls(walls_polar: np.ndarray, num_azimuth: int = 12, num_clino: int = 6) -> np.ndarray:
    if len(walls_polar) == 0:
        return np.empty((0, 3))
        
    az_sectors = np.linspace(0, 360, num_azimuth + 1)
    cl_sectors = np.linspace(-90, 90, num_clino + 1)
    filtered_shots = []
    
    for i in range(num_azimuth):
        mask_az = (walls_polar[:, 1] >= az_sectors[i]) & (walls_polar[:, 1] < az_sectors[i+1])
        az_pts = walls_polar[mask_az]
        
        if len(az_pts) == 0:
            continue
            
        for j in range(num_clino):
            mask_cl = (az_pts[:, 2] >= cl_sectors[j]) & (az_pts[:, 2] < cl_sectors[j+1])
            sector_pts = az_pts[mask_cl]
            
            if len(sector_pts) > 0:
                max_dist_idx = np.argmax(sector_pts[:, 0])
                filtered_shots.append(sector_pts[max_dist_idx])
                
    result = np.array(filtered_shots)
    return result[np.argsort(result[:, 1])]


def process_cloud_to_therion(
        cloud_xyz: np.ndarray, 
        cloud_gps: np.ndarray, 
        track_xyz: np.ndarray, 
        track_gps: np.ndarray, 
        scale:     np.ndarray,
        survey_name: str,
        time_weight=256.0
    ) -> TherionSurveyData:
        
    survey = TherionSurveyData(survey_name=survey_name)
    num_stations = len(track_xyz)

    track_m = track_xyz * scale
    cloud_m = cloud_xyz * scale

    t_start = track_gps[0]
    t_end = track_gps[-1]
    t_range = t_end - t_start if (t_end - t_start) > 0 else 1.0

    track_t_norm = (track_gps - t_start) / t_range
    cloud_t_norm = (cloud_gps - t_start) / t_range

    track_4d = np.column_stack((track_m, track_t_norm * time_weight))
    cloud_4d = np.column_stack((cloud_m, cloud_t_norm * time_weight))

    track_tree = KDTree(track_4d)
    
    points_to_stations_dist, station_indices = track_tree.query(
        cloud_4d, 
        distance_upper_bound=30 / scale[0] + time_weight
    )
    
    for j in range(num_stations - 1):
        p1, p2 = track_xyz[j], track_xyz[j+1]
        mag_vector = ((p2 - p1) * scale).reshape(1, 3)
        mag_polar = cartesian_to_polar(mag_vector)[0]
        survey.add_centerline_record(
            s_from=j, s_to=j + 1, 
            length=float(mag_polar[0]), 
            compass=float(mag_polar[1]), 
            clino=float(mag_polar[2])
        )

    for j in range(num_stations):
        mask = (station_indices == j)
        if not np.any(mask):
            continue
            
        local_pts = cloud_xyz[mask]
        p_station = track_xyz[j]
        
        vectors_slice = (local_pts - p_station) * scale
        walls_polar = cartesian_to_polar(vectors_slice)
        
        cleaned_walls = sort_walls(walls_polar, num_azimuth=16, num_clino=8)
        
        for shot in cleaned_walls:
            survey.add_wall_shot(
                s_from=j, 
                length=float(shot[0]), 
                compass=float(shot[1]), 
                clino=float(shot[2])
            )
        
    return survey
