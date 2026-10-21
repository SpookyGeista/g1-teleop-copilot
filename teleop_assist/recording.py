"""Lossless training/replay archive, with fast bounded CPU compression."""
import zipfile
import numpy as np

def write_rgbd(path,frame,compression=1):
    k=frame['intrinsics']
    matrix=np.asarray([[k['fx'],0,k['cx']],[0,k['fy'],k['cy']],[0,0,1]]) if isinstance(k,dict) else np.asarray(k)
    arrays={'rgb':frame['rgb'],'depth':frame['depth'],'intrinsics':matrix,
            'T_world_camera':frame['T_world_camera'],'timestamp':frame['timestamp'],
            'frame_id':frame['frame_id'],'sim_time':frame.get('sim_time',0.)}
    if compression not in (0,1):raise ValueError('RGB-D compression must be 0 or 1')
    with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_DEFLATED if compression else zipfile.ZIP_STORED,
                         compresslevel=1 if compression else None) as archive:
        for name,value in arrays.items():
            with archive.open(name+'.npy','w',force_zip64=True) as entry:
                np.lib.format.write_array(entry,np.asarray(value),allow_pickle=False)
