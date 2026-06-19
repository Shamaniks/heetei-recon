"""
Dispatcher of utils module
Usage:
    command: python -m restore_intensity 
        --cloud <file.laz|las|npz> 
        --track <file.npz> 
        [--intensity <file.npz>] 
        [--output <file.npz>]
    note: 
        --intensity REQUIRED in case --cloud is .npz (las files contain intensity already)
        --output default is `output.npz`
        
"""


from utils.ingest import LazReader


def restore_intensity_dispatcher(
        cloud_path:     str,
        track_path:     str,
        output_path:    str,
        cfg:            dict,
        intensity_path: str = None
    ) -> None:
    # FIXME .npz reading isn't implemented
    # FIXME saving isn't implemented

    with LazReader(cloud_path) as lr:
        xyz_cloud = lr.get_xyz()
        intensity = lr.get_intensity()

    with LazReader(track_path) as lr:
        xyz_track = lr.get_xyz()

    from utils.restore_intensity import restore_intensity

    restore_intensity(
        xyz_cloud,
        xyz_track,
        intensity,
        cfg["chunk_size"]
    )
    
