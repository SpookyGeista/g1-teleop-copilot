"""Read the reference declared by the actually retained Quest sender.

Inspect its calibration assignment without importing CloudXR or starting a
second robot controller. A changed/unknown reference fails explicitly.
"""
import ast
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation

SOURCE = Path('data/teleop_archives/g1_pose_2026-09-09/gear_sonic/scripts/pico_manager_thread_server.py')


def zero_reference_root_height(urdf, support_top, clearance=.003):
    """Place zero-pose foot collision spheres above the actual fixture plate."""
    root = ET.parse(urdf).getroot()
    transforms = {'pelvis':np.eye(4)}
    pending = list(root.findall('joint'))
    while pending:
        progressed = []
        for joint in pending:
            parent = joint.find('parent').get('link')
            if parent not in transforms:
                continue
            origin = joint.find('origin'); transform = np.eye(4)
            if origin is not None:
                transform[:3,3] = np.fromstring(origin.get('xyz','0 0 0'), sep=' ')
                transform[:3,:3] = Rotation.from_euler('xyz', np.fromstring(origin.get('rpy','0 0 0'),sep=' ')).as_matrix()
            transforms[joint.find('child').get('link')] = transforms[parent]@transform
            progressed.append(joint)
        if not progressed:
            raise ValueError('URDF hierarchy cannot be resolved from pelvis')
        pending = [joint for joint in pending if joint not in progressed]
    bottoms = []
    for side in ('left','right'):
        name = side+'_ankle_roll_link'
        link = next(link for link in root.findall('link') if link.get('name') == name)
        spheres = []
        for collision in link.findall('collision'):
            sphere = collision.find('geometry/sphere')
            if sphere is not None:
                xyz = np.fromstring(collision.find('origin').get('xyz','0 0 0'),sep=' ')
                center = transforms[name]@np.r_[xyz,1.]
                spheres.append(float(center[2])-float(sphere.get('radius')))
        if len(spheres) != 4:
            raise ValueError('Expected the retained G1 four-sphere foot collision model')
        bottoms.extend(spheres)
    return float(support_top)+float(clearance)-min(bottoms)


def retained_zero_reference(repo, urdf):
    source = Path(repo)/SOURCE
    raw = source.read_bytes()
    candidates = []
    for method in ast.walk(ast.parse(raw)):
        if isinstance(method, ast.FunctionDef) and method.name == 'calibrate_now':
            for node in ast.walk(method):
                if (isinstance(node, ast.Assign)
                        and any(isinstance(t, ast.Attribute) and t.attr == '_override_robot_q' for t in node.targets)):
                    value = node.value
                    if (isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute)
                            and isinstance(value.func.value, ast.Name) and value.func.value.id == 'np'
                            and value.func.attr == 'zeros' and len(value.args) == 1
                            and isinstance(value.args[0], ast.Constant) and value.args[0].value == 29):
                        candidates.append(node.lineno)
                    else:
                        raise ValueError('Retained calibration no longer declares the supported 29-joint zero reference')
    if len(candidates) != 1:
        raise ValueError('Expected exactly one retained Quest calibration reference')
    revolute = [j.attrib['name'] for j in ET.parse(urdf).getroot().findall('joint') if j.attrib['type'] == 'revolute']
    body = [name for name in revolute if '_hand_' not in name]
    if len(body) != 29:
        raise ValueError('Retained reference requires the same named 29-joint G1 body')
    return {'id':'retained_quest_zero_29', 'source':str(source), 'source_line':candidates[0],
            'source_sha256':hashlib.sha256(raw).hexdigest(),
            'joint_positions':{name:0. for name in body},
            'finger_positions':{name:0. for name in revolute if '_hand_' in name},
            'scope':'Exact retained body-joint calibration reference; open fingers for this fixture'}
