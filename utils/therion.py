import numpy as np
from dataclasses import dataclass, field
from typing import List

from downsampling.track_downsampling import track_downsampling


@dataclass
class TherionSurveyData:
    survey_name: str = "survey"
    declination: float = 0.0

    records: List[str] = field(default_factory=list)


    def add_record(self, s_from: int, s_to: int, length: float, 
                   compass: float, clino: float, 
                   left: float, right: float, up: float, down: float):
        line = (f"    {s_from:<5} {s_to:<5} "
                f"{length:.2f} {compass:.1f} {clino:+.1f} "
                f"{left:.2f} {right:.2f} {up:.2f} {down:.2f}")
        self.records.append(line)


    def serialize(self) -> str:
        header = (
            f"survey {self.survey_name}\n"
            "  centerline\n"
            "    units length meters\n"
            "    units compass degrees\n"
            "    units clino degrees\n\n"
            f"    declination {self.declination:.2f} degrees\n\n"
            "    data surveying from to length compass clino left right up down\n"
        )
        footer = "\n  endcenterline\nendsurvey\n"
        return header + "\n".join(self.records) + footer


    def save(self):
        file_path = f"{self.survey_name}.th"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(self.serialize())
        print(f"[therion] survey saved: {file_path}")


def compute_step_geometry(
        p1: np.ndarray, 
        p2: np.ndarray, 
        scale: float
    ) -> tuple:
    v = (p2 - p1) * scale
    length = np.linalg.norm(v)
    
    if length < 1e-4:
        return None
        
    compass = np.degrees(np.arctan2(v[0], v[1])) % 360
    clino = np.degrees(np.arcsin(v[2] / length))
    
    n = v / length
    z_world = np.array([0.0, 0.0, 1.0])
    right_vec = np.cross(n, z_world)
    r_norm = np.linalg.norm(right_vec)
    
    right_vec = np.array([1.0, 0.0, 0.0]) if r_norm < 1e-5 else right_vec / r_norm
    up_vec = np.cross(right_vec, n)
    
    return length, compass, clino, right_vec, up_vec


def extract_wall_dimensions(
        p1: np.ndarray, 
        local_pts: np.ndarray, 
        right_vec: np.ndarray, 
        up_vec: np.ndarray,
        scale: float
    ) -> tuple:
    if len(local_pts) == 0:
        return 1.0, 1.0, 1.0, 1.0
        
    scale_f32 = np.float32(scale)
    centered_pts = (local_pts - p1).astype(np.float32, copy=False) * scale_f32
    
    proj_up = np.dot(centered_pts, up_vec)
    proj_right = np.dot(centered_pts, right_vec)
    
    up = np.max(proj_up, initial=1.0)
    down = abs(np.min(proj_up, initial=-1.0))
    right = np.max(proj_right, initial=1.0)
    left = abs(np.min(proj_right, initial=-1.0))
    
    return left, right, up, down


def process_cloud_to_therion(
        track_xyz: np.ndarray, 
        track_gps: np.ndarray, 
        cloud_xyz: np.ndarray, 
        cloud_gps: np.ndarray, 
        scale,
        survey_name: str,
        step_meters=3.0, 
        dt_sec=0.5, 
    ) -> TherionSurveyData:
    """
    Parameters
    ----------
    track_xyz:   np.ndarray (N, 3),
    track_gps:   np.ndarray (N,),   
    cloud_xyz:   np.ndarray (M, 3), 
    cloud_gps:   np.ndarray (M,),   
    scale:       float
    survey_name: str
    step_meters: float
    dt_sec:      float

    Returns 
    -------
    TherionSurveyData
    """
    cloud_sort_idx = np.argsort(cloud_gps)
    cloud_xyz = np.ascontiguousarray(cloud_xyz[cloud_sort_idx])
    cloud_gps = np.ascontiguousarray(cloud_gps[cloud_sort_idx])

    station_indices = track_downsampling(track_xyz, step_meters)
    st_xyz = track_xyz[station_indices]
    st_gps = track_gps[station_indices]
    num_stations = len(st_xyz)
    
    left_boundaries = np.searchsorted(cloud_gps, st_gps - dt_sec, side='left')
    right_boundaries = np.searchsorted(cloud_gps, st_gps + dt_sec, side='right')

    survey = TherionSurveyData(survey_name=survey_name)
    
    for j in range(num_stations - 1):
        p1, p2 = st_xyz[j], st_xyz[j+1]
        
        geo_data = compute_step_geometry(p1, p2, scale)
        if geo_data is None: continue
        length, compass, clino, right_vec, up_vec = geo_data
        
        start_idx, end_idx = left_boundaries[j], right_boundaries[j]
        local_pts = cloud_xyz[start_idx:end_idx]
        
        left, right, up, down = extract_wall_dimensions(
            p1, 
            local_pts, 
            right_vec, 
            up_vec,
            scale
        )

        survey.add_record(
            s_from=j + 1, 
            s_to=j + 2, 
            length=length, 
            compass=compass, 
            clino=clino,
            left=left, 
            right=right, 
            up=up, 
            down=down
        )
    return survey
