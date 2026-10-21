"""Build and check the right Dex3 side-entry three-finger descriptor, entirely offline.

This produces a geometry-conditioned grasp hypothesis, not a hardware grasp
certificate. No torch, DDS, network, or simulator GPU is imported.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np
if '--mujoco-worker' not in sys.argv:
    from scipy.spatial.transform import Rotation
    import trimesh

PROJECT = Path(__file__).resolve().parents[2]
SOURCE = PROJECT/'research/masters_20261003/ai_assistance/dex3_descriptor'
DEFAULT_ASSETS = PROJECT/'teleop_assist/copilot/assets/dex3_right'
WRIST = 'right_wrist_yaw_link'
PAD_LINKS = ['right_hand_thumb_2_link','right_hand_index_1_link','right_hand_middle_1_link']
PREFIX = 'right_hand_'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, data):
    Path(path).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def transform(element):
    node=element.find('origin')
    t=np.eye(4)
    if node is not None:
        t[:3,:3]=Rotation.from_euler('xyz',np.fromstring(node.get('rpy','0 0 0'),sep=' ')).as_matrix()
        t[:3,3]=np.fromstring(node.get('xyz','0 0 0'),sep=' ')
    return t


def fk(root, q):
    result={WRIST:np.eye(4)};pending=list(root.findall('joint'))
    while pending:
        changed=False
        for joint in pending[:]:
            parent=joint.find('parent').get('link')
            if parent not in result:continue
            t=transform(joint)
            if joint.get('type')=='revolute':
                r=np.eye(4)
                r[:3,:3]=Rotation.from_rotvec(np.fromstring(joint.find('axis').get('xyz'),sep=' ')*q[joint.get('name')]).as_matrix()
                t=t@r
            result[joint.find('child').get('link')]=result[parent]@t
            pending.remove(joint);changed=True
        if not changed:raise ValueError('URDF is not a connected hand rooted at wrist')
    return result


def interpolate(open_q, close_q, fraction):
    return {n:(1-fraction)*open_q[n]+fraction*close_q[n] for n in open_q}


def canonical_transform():
    # Side-entry grasp: +Z_C=+Y_wrist, +X_C=-X_wrist, +Y_C=+Z_wrist.
    # Place the canonical origin at x=140 mm along the wrist, near the
    # observed closure aperture. This is a declared design origin, not a TCP.
    b=np.eye(4)
    b[:3,:3]=[[-1,0,0],[0,0,1],[0,1,0]]
    b[:3,3]=[.140,0,0]
    return b


def make_config(root):
    pres=json.loads((SOURCE/'presets.json').read_text())['presets']
    opening=dict(pres['open']);closing=dict(pres['close'])
    # The existing research synergy's +0.4 thumb abduction shifts the thumb
    # to one finger. Lock its abduction at zero to oppose the two-finger pair.
    opening[PREFIX+'thumb_0_joint']=closing[PREFIX+'thumb_0_joint']=0.
    return {
        'open':opening,'close':closing,
        'fingertip':[0.,0.,.065],
        'sweep_volume':{'extents':[.100,.060,.030],'offset':[0.,0.,.065],
                        'extents2':[.040,.060,.020],'offset2':[0.,0.,.065]},
        'links':[n.get('name') for n in root.findall('link')],
        'standoff':[0.,.010],'symmetric':False,'type':'revolute_3f',
        'base_rotation':canonical_transform().tolist(),
        'descriptor_schema':'punktnull.dex3_right.side_entry.v1',
        'status':'geometry_checked_design_hypothesis_not_grasp_success_certificate',
        'canonical_frame':'C; +X=-wristX, +Y=wristZ, +Z=wristY, origin_wrist=[0.140,0,0]m',
        'pose_output_contract':'T_pointcloud_canonical; T_world_XR_R_ee=T_world_pointcloud @ grasp @ base_rotation @ Trans(0.05,0,0)',
        'fingertip_semantics':'Named for upstream schema compatibility: open inner-box center, not a measured robot TCP.',
        'closure_semantics':'Named joint targets; real-world DDS order is not certified; source abduction changed to zero for symmetric opposition.',
        'operating_scope':'One right-hand side-entry grasp family for rigid objects long enough to span index and middle pads. Human retains fingers until contact-aware execution is independently validated.',
    }


def load_meshes(root, asset_dir):
    meshes={}
    for link in root.findall('link'):
        parts=[]
        for visual in link.findall('visual'):
            node=visual.find('geometry/mesh')
            if node is None:continue
            mesh=trimesh.load(asset_dir/node.get('filename'),force='mesh',process=True)
            mesh.apply_scale(np.fromstring(node.get('scale','1 1 1'),sep=' '))
            mesh.apply_transform(transform(visual));parts.append(mesh)
        if parts:meshes[link.get('name')]=trimesh.util.concatenate(parts)
    return meshes


def world_meshes(root, meshes, q, b):
    frames=fk(root,q);out={}
    for name,mesh in meshes.items():
        m=mesh.copy();m.apply_transform(b@frames[name]);out[name]=m
    return out


def triangle_box_overlap(triangles, center, extents):
    """Exact separating-axis triangle/AABB boundary intersection, within eps.

    SAT axes: three box axes, triangle normal and nine edge/box-axis crosses.
    This tests the represented triangles, not their convex hulls.
    """
    t=np.asarray(triangles,float)-np.asarray(center,float)
    half=np.asarray(extents,float)/2
    active=np.all(t.min(1)<=half+1e-10,axis=1)&np.all(t.max(1)>=-half-1e-10,axis=1)
    indices=np.flatnonzero(active)
    if not len(indices):return indices
    t=t[indices]
    edges=[t[:,1]-t[:,0],t[:,2]-t[:,1],t[:,0]-t[:,2]]
    valid=np.ones(len(t),dtype=bool)
    axis=np.cross(edges[0],edges[1]);p=np.einsum('nvc,nc->nv',t,axis);r=np.abs(axis)@half
    valid &= (p.min(1)<=r+1e-10)&(p.max(1)>=-r-1e-10)
    for edge in edges:
        for basis in np.eye(3):
            axis=np.cross(edge,basis);p=np.einsum('nvc,nc->nv',t,axis);r=np.abs(axis)@half
            valid &= (p.min(1)<=r+1e-10)&(p.max(1)>=-r-1e-10)
    return indices[valid]


def check_inner_box(meshes, center, extents):
    details={}
    for name,mesh in meshes.items():
        hits=triangle_box_overlap(mesh.triangles,center,extents)
        # SAT alone misses a box entirely inside a solid closed mesh.
        outside_aabb=bool(np.any(np.asarray(center)<mesh.bounds[0]) or np.any(np.asarray(center)>mesh.bounds[1]))
        inside=False if outside_aabb else (bool(mesh.contains(np.asarray(center)[None])[0]) if mesh.is_watertight else None)
        details[name]={'intersecting_triangles':int(len(hits)),'center_inside_mesh':inside,
                       'center_outside_mesh_aabb':outside_aabb,'watertight':bool(mesh.is_watertight)}
    return {'passed':all(v['intersecting_triangles']==0 and v['center_inside_mesh'] is False for v in details.values()),
            'scope':'Exact mesh triangle/AABB boundary SAT and mesh ray parity at box center; uses visual geometry copied from actual Dex3 assets.',
            'links':details}


def contact_patches(meshes):
    patches={}
    for name in PAD_LINKS:
        mesh=meshes[name];centers=mesh.triangles_center;normals=mesh.face_normals
        is_thumb='thumb' in name
        length_axis=1 if is_thumb else 0
        normal_axis=0 if is_thumb else 1
        # Broad compliant-looking distal face: exact reproducible ROI on the
        # actual shell. Not a manufacturer-confirmed tactile/contact sensor.
        mask=(centers[:,length_axis]>=.015)&(centers[:,length_axis]<=.045)&(normals[:,normal_axis]>.8)
        selected=np.flatnonzero(mask)
        if len(selected)<20:raise ValueError(f'No meaningful distal patch for {name}')
        weights=mesh.area_faces[selected]
        patches[name]={'frame':name,'triangle_ids':selected.tolist(),
                       'center_local_m':np.average(centers[selected],axis=0,weights=weights).tolist(),
                       'area_m2':float(weights.sum()),
                       'selection':{'longitudinal_axis':length_axis,'range_m':[.015,.045],
                                    'outward_normal_axis':normal_axis,'min_component':.8},
                       'status':'engineering contact-face annotation; not a force/tactile measurement'}
    return patches


def render_qa(root, meshes, config, patches, output):
    os.environ.setdefault('MPLCONFIGDIR','/tmp/punktnull-mpl-descriptor')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig,axes=plt.subplots(3,3,figsize=(15,13),sharex='col',sharey='col')
    b=np.asarray(config['base_rotation']);sv=config['sweep_volume']
    states=[('open',config['open'],'extents','offset'),
            ('mid',interpolate(config['open'],config['close'],.5),'extents2','offset2'),
            ('close',config['close'],None,None)]
    colors={'thumb':'#c73344','index':'#087bbd','middle':'#29955f','palm':'#a1a1a1'}
    for row,(label,q,ek,ok) in enumerate(states):
        frames=fk(root,q);posed=world_meshes(root,meshes,q,b)
        for name,mesh in posed.items():
            v=np.asarray(mesh.vertices);v=v[::max(1,len(v)//1500)]*1000
            color=next((c for k,c in colors.items() if k in name),'#9e73ba')
            for col,(a,d) in enumerate([(0,2),(0,1),(2,1)]):
                axes[row,col].scatter(v[:,a],v[:,d],s=.4,c=color,alpha=.5)
            if name in patches:
                patch=patches[name];t=b@frames[name]
                p=np.asarray(patch['center_local_m'])@t[:3,:3].T+t[:3,3]
                pts=meshes[name].triangles_center[patch['triangle_ids']][::40]
                pts=(pts@t[:3,:3].T+t[:3,3])*1000
                for col,(a,d) in enumerate([(0,2),(0,1),(2,1)]):
                    axes[row,col].scatter(pts[:,a],pts[:,d],s=2,c='#ff9d00',alpha=.8)
                    axes[row,col].scatter(p[a]*1000,p[d]*1000,s=35,c='black',marker='x')
        for col,(a,d) in enumerate([(0,2),(0,1),(2,1)]):
            ax=axes[row,col]
            if ek:
                center=np.array(sv[ok])*1000;extent=np.array(sv[ek])*1000
                ax.add_patch(Rectangle((center[a]-extent[a]/2,center[d]-extent[d]/2),extent[a],extent[d],fill=False,lw=2,color='#8431bd'))
            ax.set_title(label+' | '+['closing/approach','closing/finger span','approach/finger span'][col])
            ax.set_xlabel('XYZ'[a]+' canonical (mm)');ax.set_ylabel('XYZ'[d]+' canonical (mm)')
            ax.set_aspect('equal');ax.grid(alpha=.2)
    fig.suptitle('Dex3 right: side-entry design | orange = annotated distal faces | purple = inner-box hypothesis\nCanonical +Z = wrist +Y; +X = wrist -X. Thumb abduction fixed at 0. Not a stable-grasp certificate.',fontsize=12)
    fig.tight_layout(rect=[0,0,1,.95]);fig.savefig(output/'descriptor_orthographic.png',dpi=150);plt.close(fig)


def build_mjcf(root, asset_dir):
    model=ET.Element('mujoco',{'model':'dex3_descriptor_contact_probe'})
    ET.SubElement(model,'compiler',{'angle':'radian','autolimits':'true'})
    ET.SubElement(model,'option',{'gravity':'0 0 0','timestep':'0.002'})
    assets=ET.SubElement(model,'asset');world=ET.SubElement(model,'worldbody')
    meshes={};links={l.get('name'):l for l in root.findall('link')}
    for link in links.values():
        for m in link.findall('visual/geometry/mesh'):
            filename=m.get('filename');name=Path(filename).stem
            if name not in meshes:
                meshes[name]=filename
                ET.SubElement(assets,'mesh',{'name':name,'file':str((asset_dir/filename).resolve())})
    def vec(x):return ' '.join(str(float(v)) for v in x)
    def add_link(name,parent,urdf_joint=None):
        bodyattrs={'name':name}
        if urdf_joint is not None:
            t=transform(urdf_joint);bodyattrs['pos']=vec(t[:3,3]);bodyattrs['quat']=vec(Rotation.from_matrix(t[:3,:3]).as_quat()[[3,0,1,2]])
        body=ET.SubElement(parent,'body',bodyattrs)
        if urdf_joint is not None and urdf_joint.get('type')=='revolute':
            lim=urdf_joint.find('limit')
            ET.SubElement(body,'joint',{'name':urdf_joint.get('name'),'type':'hinge','axis':urdf_joint.find('axis').get('xyz'),
                                       'range':lim.get('lower')+' '+lim.get('upper'),'damping':'1'})
        for visual in links[name].findall('visual'):
            m=visual.find('geometry/mesh')
            if m is None:continue
            t=transform(visual)
            ET.SubElement(body,'geom',{'name':name+'_mesh','type':'mesh','mesh':Path(m.get('filename')).stem,
                                      'pos':vec(t[:3,3]),'quat':vec(Rotation.from_matrix(t[:3,:3]).as_quat()[[3,0,1,2]]),
                                      'contype':'1','conaffinity':'2','mass':'0.02','margin':'0'})
        for joint in root.findall('joint'):
            if joint.find('parent').get('link')==name:add_link(joint.find('child').get('link'),body,joint)
    add_link(WRIST,world)
    ET.SubElement(world,'geom',{'name':'probe','type':'box','size':'.01 .01 .01','pos':'.5 .5 .5',
                               'contype':'2','conaffinity':'1','margin':'0','rgba':'1 0.6 0 0.5'})
    ET.indent(model)
    return model


def mujoco_probe(asset_dir, output, config):
    python=Path('/home/loya2/miniconda3/envs/mujoco/bin/python')
    if not python.exists():return {'status':'unavailable','reason':'CPU MuJoCo interpreter absent'}
    result=subprocess.run([str(python),str(Path(__file__).resolve()),'--mujoco-worker',str(asset_dir),str(output)],
                          text=True,capture_output=True,timeout=120)
    (output/'mujoco_worker.log').write_text(result.stdout+'\n'+result.stderr)
    if result.returncode:raise RuntimeError('CPU MuJoCo probe failed: '+result.stderr[-1500:])
    return json.loads((output/'mujoco_contacts.json').read_text())


def run_mujoco_worker(asset_dir,output):
    # Loaded in the existing CPU MuJoCo environment. This branch is dispatched
    # before the heavy analysis imports by a small script wrapper below.
    import mujoco
    config=json.loads((asset_dir/'config.json').read_text())
    model=mujoco.MjModel.from_xml_path(str(output/'contact_probe.xml'));data=mujoco.MjData(model)
    gid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,'probe')
    # Deliberately conservative broad-phase bounds while shape/size are
    # varied. Narrow-phase collision still uses the exact selected primitive.
    model.geom_rbound[gid]=.50
    model.geom_aabb[gid]=[0,0,0,.25,.25,.25]
    joints={n:model.jnt_qposadr[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,n)] for n in config['open']}
    invb=np.linalg.inv(np.asarray(config['base_rotation']))
    def contacts(q):
        for n,v in q.items():data.qpos[joints[n]]=v
        mujoco.mj_forward(model,data)
        rows=[]
        for c in data.contact:
            ids=[int(c.geom1),int(c.geom2)]
            if gid in ids:
                other=ids[1] if ids[0]==gid else ids[0]
                rows.append({'link':mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,other).removesuffix('_mesh'),
                             'distance_m':float(c.dist)})
        return rows
    model.geom_pos[gid]=[.09,0,0];model.geom_size[gid]=[.015,.015,.015]
    positive_control=contacts(config['open'])
    model.geom_pos[gid]=[.5,.5,.5]
    negative_control=contacts(config['open'])
    if not positive_control or negative_control:raise ValueError('MuJoCo collision contact controls failed')
    checks={}
    for label,f,ek,ok in [('open',0.,'extents','offset'),('mid',.5,'extents2','offset2')]:
        sv=config['sweep_volume'];model.geom_type[gid]=mujoco.mjtGeom.mjGEOM_BOX
        model.geom_size[gid]=np.asarray(sv[ek])/2
        model.geom_pos[gid]=(invb@np.r_[sv[ok],1])[:3]
        quat=np.zeros(4)
        mujoco.mju_mat2Quat(quat,invb[:3,:3].copy().reshape(-1))
        model.geom_quat[gid]=quat
        c=contacts(interpolate(config['open'],config['close'],f))
        checks[label]={'passed':len(c)==0,'contacts':c}
    # Kinematic contact feasibility only: fixed cylinder, closure sampled.
    # Convex hull collision cannot certify force closure or physical stability.
    model.geom_type[gid]=mujoco.mjtGeom.mjGEOM_CYLINDER;model.geom_quat[gid]=[1,0,0,0]
    trials=[]
    for radius in [.012,.016,.020]:
        for x in [.125,.135,.145]:
            for y in [.050,.060,.070]:
                model.geom_size[gid]=[radius,.040,0];model.geom_pos[gid]=[x,y,0]
                onset={};overlap=None;open_contacts=None
                for f in np.linspace(0,1,101):
                    c=contacts(interpolate(config['open'],config['close'],float(f)))
                    if f==0:open_contacts=c
                    touched={v['link'] for v in c}
                    for link in PAD_LINKS:
                        if link in touched and link not in onset:onset[link]=float(f)
                    if all(link in touched for link in PAD_LINKS) and overlap is None:
                        overlap={'closure':float(f),'max_penetration_m':max([-v['distance_m'] for v in c]+[0.]),
                                 'contacts':c}
                trials.append({'radius_m':radius,'half_length_m':.040,'center_wrist_m':[x,y,0],
                               'open_collision_free':not open_contacts,'first_distal_contacts':onset,
                               'first_simultaneous_three_distal_contact':overlap})
    result={'status':'complete','mujoco_version':mujoco.__version__,
            'method':'CPU kinematic mj_forward contacts against convex hulls of actual mesh geometry; hand self-collisions disabled to isolate probe contacts.',
            'controls':{'box_inside_palm_detected':bool(positive_control),'far_box_collision_free':not negative_control,
                        'positive_contacts':positive_control},
            'inner_boxes':checks,'shape_trials':trials,
            'shape_limitations':'Fixed objects and imposed joints can interpenetrate. These contact onsets are geometry probes, not force closure, successful grasp, dynamic execution, or a hardware result.'}
    save_json(output/'mujoco_contacts.json',result)


def check_actual_mesh_contact_example(root,meshes,config,cpu,output):
    """Recheck the best convex-hull contact example against actual CAD surfaces.

    The analytic finite-cylinder signed distance is evaluated on all vertices,
    so negative values prove real mesh surface points enter the fixed object.
    This deliberately retains imposed overlap and makes no stability claim.
    """
    available=[t for t in cpu['shape_trials'] if t['first_simultaneous_three_distal_contact']]
    if not available:return {'status':'no_convex_hull_candidate'}
    best=min(available,key=lambda t:t['first_simultaneous_three_distal_contact']['max_penetration_m'])
    f=best['first_simultaneous_three_distal_contact']['closure'];q=interpolate(config['open'],config['close'],f)
    posed=world_meshes(root,meshes,q,np.eye(4));center=np.array(best['center_wrist_m'])
    r=best['radius_m'];half=best['half_length_m'];checks={}
    for name in PAD_LINKS:
        v=posed[name].vertices-center
        d=np.column_stack([np.linalg.norm(v[:,:2],axis=1)-r,np.abs(v[:,2])-half])
        sdf=np.linalg.norm(np.maximum(d,0),axis=1)+np.minimum(np.max(d,axis=1),0)
        checks[name]={'minimum_surface_vertex_signed_distance_m':float(sdf.min()),
                      'vertices_strictly_inside_fixed_cylinder':int(np.sum(sdf<0))}
    result={'status':'complete','selected_static_example':best,'actual_mesh_surfaces':checks,
            'all_three_actual_mesh_surfaces_enter_fixed_cylinder':all(v['vertices_strictly_inside_fixed_cylinder']>0 for v in checks.values()),
            'limitation':'Kinematic imposed-joint test against a fixed analytical cylinder. Negative distance is overlap, not evidence of force closure or stable object retention.'}
    save_json(output/'actual_mesh_contact_example.json',result)
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4.5));b=np.array(config['base_rotation'])
    theta=np.linspace(0,2*np.pi,90);z=np.linspace(-half,half,20)
    cyl=np.array([[r*np.cos(t),r*np.sin(t),h] for t in theta for h in z])+center
    cyl=(cyl@b[:3,:3].T+b[:3,3])*1000
    for name,mesh in posed.items():
        vertices=mesh.vertices[::max(1,len(mesh.vertices)//1200)]
        points=(vertices@b[:3,:3].T+b[:3,3])*1000
        color='#c73344' if 'thumb' in name else '#087bbd' if 'index' in name else '#29955f' if 'middle' in name else '#aaa'
        for ax,(a,d) in zip(axes,[(0,2),(0,1),(2,1)]):ax.scatter(points[:,a],points[:,d],s=.5,c=color,alpha=.5)
    for ax,(a,d) in zip(axes,[(0,2),(0,1),(2,1)]):
        ax.scatter(cyl[:,a],cyl[:,d],s=.3,c='#db8800',alpha=.25)
        ax.set(xlabel='XYZ'[a]+' canonical (mm)',ylabel='XYZ'[d]+' canonical (mm)');ax.axis('equal');ax.grid(alpha=.2)
    fig.suptitle(f'Actual Dex3 surfaces vs fixed cylinder: diameter {2*r*1000:.0f} mm, closure {f:.2f}\nKinematic overlap probe only; all three mesh surfaces inside object: {result["all_three_actual_mesh_surfaces_enter_fixed_cylinder"]}',fontsize=11)
    fig.tight_layout(rect=[0,0,1,.88]);fig.savefig(output/'contact_example.png',dpi=150);plt.close(fig)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--assets',type=Path,default=DEFAULT_ASSETS)
    parser.add_argument('--output',type=Path,default=PROJECT/'runs/copilot_descriptor_002')
    args=parser.parse_args();assets=args.assets.resolve();out=args.output.resolve()
    assets.mkdir(parents=True,exist_ok=True);out.mkdir(parents=True,exist_ok=True)
    for name in ['gripper.urdf','UNITREE_ASSETS_LICENSE.txt']:
        shutil.copyfile(SOURCE/name,assets/name)
    shutil.copytree(SOURCE/'meshes',assets/'meshes',dirs_exist_ok=True)
    root=ET.parse(assets/'gripper.urdf').getroot();config=make_config(root)
    meshes=load_meshes(root,assets);patches=contact_patches(meshes);b=np.asarray(config['base_rotation'])
    qmid=interpolate(config['open'],config['close'],.5)
    open_meshes=world_meshes(root,meshes,config['open'],b)
    verts=np.concatenate([m.vertices for m in open_meshes.values()]);config['bbox']=[verts.min(0).tolist(),verts.max(0).tolist()]
    config['geometry_assets_sha256']={n:digest(assets/n) for n in ['gripper.urdf']}
    save_json(assets/'config.json',config);save_json(assets/'contact_patches.json',patches)
    trimesh.util.concatenate(list(open_meshes.values())).export(assets/'vis_mesh.obj')
    # The runtime requires a canonical collision visualization mesh too. This
    # exact copied/open merged surface is useful for external collision tests;
    # no success label is inferred from its existence.
    trimesh.util.concatenate(list(open_meshes.values())).export(assets/'coll_mesh.obj')
    sv=config['sweep_volume']
    boxes={'open':check_inner_box(open_meshes,sv['offset'],sv['extents']),
           'mid':check_inner_box(world_meshes(root,meshes,qmid,b),sv['offset2'],sv['extents2'])}
    limits={j.get('name'):{k:float(v) for k,v in j.find('limit').attrib.items()} for j in root.findall('joint') if j.get('type')=='revolute'}
    limits_ok=all(limits[n]['lower']<=v<=limits[n]['upper'] for q in [config['open'],qmid,config['close']] for n,v in q.items())
    render_qa(root,meshes,config,patches,out)
    ET.ElementTree(build_mjcf(root,assets)).write(out/'contact_probe.xml',encoding='unicode')
    cpu=mujoco_probe(assets,out,config)
    surface_contact=check_actual_mesh_contact_example(root,meshes,config,cpu,out)
    checked_assets={str(p.relative_to(assets)):digest(p) for p in sorted(assets.rglob('*'))
                    if p.is_file() and p.name!='validation.json'}
    geometry_valid=bool(limits_ok and all(v['passed'] for v in boxes.values()) and all(v['passed'] for v in cpu['inner_boxes'].values()))
    report={'validation_schema':'punktnull.copilot.descriptor_validation.v1',
            'status':'complete','descriptor':'dex3_right.side_entry.v1','assets':str(assets),
            'geometry_valid':geometry_valid,
            'geometry_valid_scope':'Named-joint limits, canonical frame, source-asset identity and annotated OPEN/MID box nonintersection. Does not certify grasp success, scene collision avoidance, hardware calibration or real-world motor ordering.',
            'asset_sha256':checked_assets,
            'config_sha256':digest(assets/'config.json'),'source':str(SOURCE),'generator_sha256':digest(__file__),
            'canonical_transform':b.tolist(),'limits_passed':limits_ok,'mesh_inner_boxes':boxes,
            'mujoco_inner_boxes':cpu.get('inner_boxes'),
            'visual_review_file':'descriptor_orthographic.png','visual_review_status':'source design reviewed in descriptor_002; current render must be retained with this build',
            'design_choices':{'grasp_family':'right-hand side-entry thumb opposing index and middle; suited to objects spanning the pair',
                              'thumb_abduction':'0 rad at both endpoints; changed deliberately from the existing 0.4 rad closure to center thumb between index/middle',
                              'open_box_wrist_bounds_m':[[.09,.05,-.03],[.19,.08,.03]],
                              'mid_box_wrist_bounds_m':[[.12,.055,-.03],[.16,.075,.03]],
                              'box_rationale':'Conservative rectangular interior crops of the visible closure aperture; free-space SAT and convex-hull contact checks, not a fitted grasp-success region.'},
            'mujoco_contact_controls':cpu.get('controls'),
            'contact_candidate_summary':{'tested_cylinders':len(cpu['shape_trials']),
                'three_distal_contact_at_under_2mm_max_overlap':sum(t['first_simultaneous_three_distal_contact'] is not None and t['first_simultaneous_three_distal_contact']['max_penetration_m']<.002 for t in cpu['shape_trials']),
                'interpretation':'Static geometry contact candidates only; imposed closure permits penetration. Full per-case evidence, including poor cases, is retained in mujoco_contacts.json.'},
            'actual_mesh_contact_example':surface_contact,
            'use_limit':'Model inference may label this a Dex3-conditioned candidate. It is not a certified executable grasp and must pass downstream full-robot reachability and collision checks.'}
    save_json(out/'report.json',report)
    if not geometry_valid:
        raise RuntimeError('Descriptor geometry check failed; inspect report and revise annotation')
    save_json(assets/'validation.json',report)
    print(json.dumps({'assets':str(assets),'report':str(out/'report.json'),'mesh_boxes_passed':True,'mujoco_boxes_passed':True,
                      'sweep_volume':sv},indent=2))


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--mujoco-worker':
        run_mujoco_worker(Path(sys.argv[2]),Path(sys.argv[3]))
    else:
        main()
