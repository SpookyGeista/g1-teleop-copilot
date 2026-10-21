"""Procedural cabinet test fixture. Geometry/truth stay inside the simulator.

The fixture exposes geometric protrusions to RGB-D perception; it does not
publish its object names or coordinates to the detector. The mounted G1 test
is deliberately separate from the floating-base SONIC mission.
"""
from __future__ import annotations

import math


def build_cabinet_scene(stage, seed=0, variant="cabinet", asset_dir=None):
    if variant not in ("cabinet", "interaction"):
        raise ValueError("Scene variant must be cabinet or interaction")
    from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdPhysics, UsdShade

    root = '/World/AssistFixture'
    UsdGeom.Xform.Define(stage, root)
    materials = {}
    palette = {
        'panel': (.38, .43, .47), 'frame': (.075, .095, .12),
        'steel': (.57, .60, .62), 'black': (.035, .044, .055),
        'blue': (.035, .28, .52), 'amber': (.88, .45, .045),
        'green': (.06, .42, .19), 'red': (.66, .045, .025),
        'floor': (.20, .22, .24), 'wall': (.39, .43, .46),
        'white': (.82, .84, .80), 'copper': (.45, .19, .08),
    }
    for name, color in palette.items():
        path = root + '/Materials/' + name
        mat = UsdShade.Material.Define(stage, path)
        shader = UsdShade.Shader.Define(stage, path + '/Surface')
        shader.CreateIdAttr('UsdPreviewSurface')
        shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(.62)
        shader.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(.15 if name == 'steel' else 0.)
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
        materials[name] = mat

    collision_boxes = []
    targets = []
    cad_objects = []
    joints = []

    def finish(shape, center, mat, collision, mass=None):
        xf = UsdGeom.Xformable(shape.GetPrim())
        xf.AddTranslateOp().Set(Gf.Vec3d(*map(float, center)))
        UsdShade.MaterialBindingAPI.Apply(shape.GetPrim()).Bind(materials[mat])
        if collision:
            UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
        if mass is not None:
            UsdPhysics.RigidBodyAPI.Apply(shape.GetPrim())
            UsdPhysics.MassAPI.Apply(shape.GetPrim()).CreateMassAttr(float(mass))
        return shape

    def box(name, center, size, mat='steel', collision=True, mass=None):
        cube = UsdGeom.Cube.Define(stage, root + '/' + name)
        cube.CreateSizeAttr(1.)
        finish(cube, center, mat, collision, mass)
        UsdGeom.Xformable(cube.GetPrim()).AddScaleOp().Set(Gf.Vec3d(*map(float, size)))
        if collision:
            collision_boxes.append({'name': name, 'center': list(center), 'size': list(size), 'dynamic': mass is not None})
        return cube

    def cylinder(name, center, radius, length, mat='steel', collision=True, mass=None, axis='X'):
        shape = UsdGeom.Cylinder.Define(stage, root + '/' + name)
        shape.CreateAxisAttr(axis)
        shape.CreateRadiusAttr(float(radius))
        shape.CreateHeightAttr(float(length))
        finish(shape, center, mat, collision, mass)
        return shape

    def target(name, center, size, kind, shape='box', mat='steel', dynamic=False):
        if shape == 'cylinder':
            prim = cylinder('Parts/' + name, center, size[1]/2, size[0], mat, mass=.08 if dynamic else None)
        else:
            prim = box('Parts/' + name, center, size, mat, mass=.08 if dynamic else None)
        targets.append({'id': name, 'kind': kind, 'prim_path': str(prim.GetPath()),
                        'center': list(center), 'size': list(size),
                        'surface_position': [center[0]-size[0]/2, center[1], center[2]],
                        'surface_normal': [-1., 0., 0.], 'dynamic': dynamic})
        cad_objects.append({'id': name, 'kind': kind, 'components': [{'shape': shape, 'size': list(size)}],
                            'actions': ['approach', 'grasp'], 'articulated': dynamic,
                            'contact_point_object_m': [-size[0]/2, 0., 0.],
                            'approach_direction_object': [1., 0., 0.]})
        return prim

    # A real floor, hall walls, ceiling beams and local area lighting. All local
    # geometry: no network assets or scenery packages are needed at startup.
    box('Hall/Floor', (0, 0, -.055), (7, 7, .10), 'floor')
    box('Hall/BackWall', (2.4, 0, 1.6), (.12, 7, 3.2), 'wall')
    box('Hall/SideWall', (0, 2.5, 1.6), (5, .12, 3.2), 'wall')
    for n, y in enumerate((-2., 0., 2.)):
        box(f'Hall/Beam{n}', (.4, y, 2.75), (4.5, .09, .15), 'frame')
    # Visible support makes the fixed pelvis assumption inspectable.
    box('PelvisMount/Base', (-.18, 0, .03), (.52, .48, .06), 'frame')
    box('PelvisMount/Post', (-.22, 0, .43), (.065, .065, .8), 'amber')
    box('PelvisMount/Brace', (-.10, 0, .68), (.26, .055, .055), 'amber')

    dome = UsdLux.DomeLight.Define(stage, root + '/Lights/Ambient')
    dome.CreateIntensityAttr(550.)
    dome.CreateColorAttr(Gf.Vec3f(.77, .84, 1.))
    for name, pos, intensity, color in (
        ('Key', (-.6, -.4, 2.5), 2200., (1., .91, .78)),
        ('Fill', (.1, 1.1, 2.), 1300., (.76, .87, 1.)),
    ):
        light = UsdLux.SphereLight.Define(stage, root + '/Lights/' + name)
        light.CreateIntensityAttr(intensity)
        light.CreateRadiusAttr(.30)
        light.CreateColorAttr(Gf.Vec3f(*color))
        UsdGeom.Xformable(light.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(*pos))

    panel_x = .60
    dy = ((int(seed) % 3)-1)*.009
    if variant == 'interaction':
        # A physical recess is necessary for the button's +X press travel.
        # A solid backplate directly behind it would block the slider even
        # though the joint allows 12 mm. Four panels leave an 80 mm opening.
        cy, cz, half = -.11+dy, 1.15, .04
        ymin, ymax, zmin, zmax = -.525, .345, .64, 1.43
        for name, y0, y1, z0, z1 in (
            ('BackplateLower', ymin, ymax, zmin, cz-half),
            ('BackplateUpper', ymin, ymax, cz+half, zmax),
            ('BackplateSideA', ymin, cy-half, cz-half, cz+half),
            ('BackplateSideB', cy+half, ymax, cz-half, cz+half),
        ):
            box('Cabinet/'+name, (panel_x+.018, (y0+y1)/2, (z0+z1)/2),
                (.036, y1-y0, z1-z0), 'panel')
        box('Cabinet/ButtonRecess', (.656, cy, cz), (.012, .08, .08), 'black')
    else:
        box('Cabinet/Backplate', (panel_x+.018, -.09, 1.035), (.036, .87, .79), 'panel')
    for name, center, size in (
        ('Top', (.64, -.09, 1.45), (.16, .96, .055)),
        ('Bottom', (.64, -.09, .61), (.16, .96, .055)),
        ('Left', (.64, .37, 1.035), (.16, .05, .84)),
        ('Right', (.64, -.55, 1.035), (.16, .05, .84)),
    ):
        box('Cabinet/' + name, center, size, 'frame')
    # Targets have meaningful 12–50 mm depth contrast and are separated in the
    # camera image. Seed changes small layout offsets, never detector labels.
    dy = ((int(seed) % 3)-1)*.009
    target('BlueSelector', (.581, -.26+dy, 1.15), (.038, .065, .065), 'rotary_selector', 'cylinder', 'blue', dynamic=variant == 'interaction')
    target('GreenButton', (.580, -.11+dy, 1.15), (.040, .058, .058), 'push_button', 'cylinder', 'green', dynamic=True)
    target('RedButton', (.583, .04+dy, 1.15), (.034, .058, .058), 'push_button', 'cylinder', 'red')
    if variant == 'cabinet':
        target('LowerSelector', (.578, -.26+dy, .985), (.044, .072, .072), 'rotary_selector', 'cylinder', 'black')
        target('LeverBase', (.580, -.10+dy, .985), (.040, .048, .092), 'lever', mat='amber')
        target('Handle', (.575, .095+dy, .96), (.050, .035, .145), 'handle', mat='steel')
    target('TerminalA', (.587, -.30+dy, .805), (.026, .065, .050), 'terminal', mat='white')
    target('TerminalB', (.587, -.18+dy, .805), (.026, .065, .050), 'terminal', mat='white')
    target('TerminalC', (.587, -.06+dy, .805), (.026, .065, .050), 'terminal', mat='blue')
    for i, (y, z) in enumerate(((-.405, 1.27), (.20, 1.27), (-.405, 1.00), (.20, .77))):
        target('Screw' + str(i), (.593, y, z), (.014, .018, .018), 'screw', 'cylinder')

    # The green button is a physically constrained sliding rigid body. Its
    # displacement is never assigned by application logic. Spring returns it.
    joint = UsdPhysics.PrismaticJoint.Define(stage, root + '/Joints/GreenButtonSlide')
    joint.CreateBody1Rel().SetTargets([Sdf.Path(root + '/Parts/GreenButton')])
    joint.CreateAxisAttr('X')
    joint.CreateLocalPos0Attr(Gf.Vec3f(.580, -.11+dy, 1.15))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0., 0., 0.))
    joint.CreateLowerLimitAttr(0.)
    joint.CreateUpperLimitAttr(.012)
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), 'linear')
    drive.CreateTypeAttr('force')
    drive.CreateTargetPositionAttr(0.)
    drive.CreateStiffnessAttr(120.)
    drive.CreateDampingAttr(3.)
    drive.CreateMaxForceAttr(2.)
    button_joint_path = str(joint.GetPath())
    joints.append({'id': 'GreenButton', 'body_path': root+'/Parts/GreenButton',
                   'joint_path': button_joint_path, 'type': 'prismatic',
                   'anchor_world': [.580, -.11+dy, 1.15], 'axis_world': [1., 0., 0.],
                   'limits': [0., .012]})
    button_asset = next(obj for obj in cad_objects if obj['id'] == 'GreenButton')
    button_asset.update({'actions': ['approach', 'push'], 'mechanical_joint': {
        'type': 'prismatic', 'axis_object': [1., 0., 0.], 'limits_m': [0., .012]}})

    if variant == 'interaction':
        # A free-to-turn selector with real revolute constraints. Its marker
        # rotates as collision geometry on the same rigid body, not a UI cue.
        marker = {'shape': 'box', 'center': [-.021, 0., .009], 'size': [.008, .012, .042]}
        box('Parts/BlueSelector/Marker', marker['center'], marker['size'], 'white')
        knob_asset = next(obj for obj in cad_objects if obj['id'] == 'BlueSelector')
        knob_asset['components'].append(marker)
        knob_asset.update({'actions': ['approach', 'grasp', 'rotate'], 'mechanical_joint': {
            'type': 'revolute', 'axis_object': [1., 0., 0.], 'origin_object_m': [0., 0., 0.],
            'limits_rad': [-math.pi/2, math.pi/2]}})
        knob = UsdPhysics.RevoluteJoint.Define(stage, root+'/Joints/BlueSelectorRotate')
        knob.CreateBody1Rel().SetTargets([Sdf.Path(root+'/Parts/BlueSelector')])
        knob.CreateAxisAttr('X')
        knob.CreateLocalPos0Attr(Gf.Vec3f(.581, -.26+dy, 1.15))
        knob.CreateLocalPos1Attr(Gf.Vec3f(0., 0., 0.))
        knob.CreateLowerLimitAttr(-90.)
        knob.CreateUpperLimitAttr(90.)
        friction = UsdPhysics.DriveAPI.Apply(knob.GetPrim(), 'angular')
        friction.CreateTypeAttr('force')
        friction.CreateStiffnessAttr(0.)
        # Angular USD drive rates are degrees/s; this is ~0.0057 N*m*s/rad.
        friction.CreateDampingAttr(.0001)
        friction.CreateMaxForceAttr(.05)
        friction.CreateTargetVelocityAttr(0.)
        joints.append({'id': 'BlueSelector', 'body_path': root+'/Parts/BlueSelector',
                       'joint_path': str(knob.GetPath()), 'type': 'revolute',
                       'anchor_world': [.581, -.26+dy, 1.15], 'axis_world': [1., 0., 0.],
                       'limits': [-math.pi/2, math.pi/2]})

        # Compound rigid drawer front and U-shaped pull handle. The open gap
        # behind the bar allows fingers to engage physically. No attachment,
        # pose teleport, or application-driven joint target is used.
        # Place the drawer in the right arm's natural work sector; a
        # centreline target induces a shoulder self-collision on retraction.
        drawer_center = [.574, -.22+dy, .955]
        body_path = root+'/Parts/PullHandle'
        drawer = UsdGeom.Xform.Define(stage, body_path)
        UsdGeom.Xformable(drawer.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(*drawer_center))
        UsdPhysics.RigidBodyAPI.Apply(drawer.GetPrim())
        UsdPhysics.MassAPI.Apply(drawer.GetPrim()).CreateMassAttr(.30)
        from .fixture_assets import drawer_handle_components, pull_handle_grasp_asset
        components = drawer_handle_components()
        for n, component in enumerate(components):
            box('Parts/PullHandle/Part'+str(n), component['center'], component['size'],
                'panel' if n == 0 else 'steel')
        slide = UsdPhysics.PrismaticJoint.Define(stage, root+'/Joints/PullHandleSlide')
        slide.CreateBody1Rel().SetTargets([Sdf.Path(body_path)])
        slide.CreateAxisAttr('X')
        slide.CreateLocalPos0Attr(Gf.Vec3f(*drawer_center))
        slide.CreateLocalPos1Attr(Gf.Vec3f(0., 0., 0.))
        slide.CreateLowerLimitAttr(-.16)
        slide.CreateUpperLimitAttr(0.)
        damping = UsdPhysics.DriveAPI.Apply(slide.GetPrim(), 'linear')
        damping.CreateTypeAttr('force')
        damping.CreateStiffnessAttr(0.)
        damping.CreateDampingAttr(2.)
        damping.CreateMaxForceAttr(.5)
        damping.CreateTargetVelocityAttr(0.)
        joints.append({'id': 'PullHandle', 'body_path': body_path,
                       'joint_path': str(slide.GetPath()), 'type': 'prismatic',
                       'anchor_world': drawer_center, 'axis_world': [1., 0., 0.],
                       'limits': [-.16, 0.]})
        cad_objects.append({'id': 'PullHandle', 'kind': 'drawer_handle',
                            'components': components, 'articulated': True,
                            'actions': ['approach', 'grasp', 'pull', 'push'],
                            'contact_point_object_m': [-.076, 0., 0.],
                            'approach_direction_object': [1., 0., 0.],
                            'mechanical_joint': {'type': 'prismatic', 'axis_object': [1., 0., 0.],
                                                 'limits_m': [-.16, 0.]},
                            'grasp_templates': [{'id': 'right_g1_bounded_pull_v1',
                                'tool_point_object_m': [-.056, 0., -.050],
                                'wrist_quaternion_object_xyzw': [math.sqrt(.5), 0., 0., math.sqrt(.5)],
                                'grip_scalar': .85, 'approach_standoff_m': .09,
                                'validated_pull_distance_m': .04,
                                'validated_pull_duration_s': 6.,
                                'validation': 'oracle PhysX G1-hand test drawer_grasp_probe_004; not perception validation',
                                'scope': 'Known CAD grasp candidate for this hand and fixture; 100mm pull was not successful'}]})
        cad_objects.append(pull_handle_grasp_asset(cad_objects[-1]['grasp_templates']))
        targets.append({'id': 'PullHandle', 'kind': 'drawer_handle', 'prim_path': body_path,
                        'center': drawer_center, 'size': [.10, .30, .18],
                        'surface_position': [drawer_center[0]-.076, drawer_center[1], drawer_center[2]],
                        'surface_normal': [-1., 0., 0.], 'dynamic': True})

    # A cluttered wiring band, recessed away from the working row. Wires have
    # actual collision geometry; their colors are not a perception code.
    for i in range(13):
        y = -.39 + i*.040
        z = 1.345 + .013*math.sin(i*.7)
        cylinder(f'Wiring/Vertical{i}', (.591, y, z), .0038, .105, ('black', 'blue', 'copper')[i % 3], axis='Z')
        cylinder(f'Wiring/Bottom{i}', (.591, y, .710+.007*math.sin(i)), .0038, .08, ('blue', 'black')[i % 2], axis='Z')
    box('Wiring/TopDuct', (.596, -.12, 1.405), (.008, .65, .025), 'frame')
    catalog = None
    if asset_dir is not None:
        from .fixture_assets import write_cad_assets
        catalog = write_cad_assets(cad_objects, asset_dir)
    return {'fixture': 'fixed_pelvis_g1_'+variant+'_v2', 'root': root,
            'targets': targets, 'collision_boxes': collision_boxes,
            'dynamic_button_joint': button_joint_path, 'fixture_joints': joints,
            'known_asset_catalog': catalog,
            'evaluation_only': True, 'panel_front_x': panel_x,
            'disclosures': ['Fixed pelvis manipulation fixture; no walking controller.',
                            ('Green button, drawer/pull handle and blue rotary selector are physically articulated.'
                             if variant == 'interaction' else 'Only the green push button is physically articulated.'),
                            'Joint motion must arise from PhysX contact; no automatic grasp attachment.',
                            'Other components are collision-active approach/grasp targets, not repair tasks.']}
