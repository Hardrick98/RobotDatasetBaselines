dataset_info = dict(
    dataset_name='RobotDataset',

    keypoint_info={
        0: dict(name='Torso', id=0, color=[0, 200, 0], type='', swap=''),

        1: dict(name='Neck', id=1, color=[0, 200, 0], type='lower', swap=''),

        # LEFT (blu)
        2: dict(name='L_Hip', id=2, color=[0, 102, 204], type='lower', swap='R_Hip'),
        3: dict(name='L_Knee', id=3, color=[0, 102, 204], type='lower', swap='R_Knee'),
        4: dict(name='L_Ankle', id=4, color=[0, 102, 204], type='lower', swap='R_Ankle'),
        5: dict(name='L_Shoulder', id=5, color=[0, 102, 204], type='upper', swap='R_Shoulder'),
        6: dict(name='L_Elbow', id=6, color=[0, 102, 204], type='upper', swap='R_Elbow'),
        7: dict(name='L_Wrist', id=7, color=[0, 102, 204], type='upper', swap='R_Wrist'),

        # RIGHT (rosso)
        8: dict(name='R_Hip', id=8, color=[220, 50, 32], type='lower', swap='L_Hip'),
        9: dict(name='R_Knee', id=9, color=[220, 50, 32], type='lower', swap='L_Knee'),
        10: dict(name='R_Ankle', id=10, color=[220, 50, 32], type='lower', swap='L_Ankle'),
        11: dict(name='R_Shoulder', id=11, color=[220, 50, 32], type='upper', swap='L_Shoulder'),
        12: dict(name='R_Elbow', id=12, color=[220, 50, 32], type='upper', swap='L_Elbow'),
        13: dict(name='R_Wrist', id=13, color=[220, 50, 32], type='upper', swap='L_Wrist'),
    },

    skeleton_info={
        # Torso / center (verde)
        3: dict(link=('L_Hip', 'R_Hip'), id=4, color=[0, 200, 0]),
        6: dict(link=('L_Shoulder', 'R_Shoulder'), id=7, color=[0, 200, 0]),

        # LEFT (blu)
        0: dict(link=('L_Knee', 'L_Hip'), id=1, color=[0, 102, 204]),
        1: dict(link=('L_Ankle', 'L_Knee'), id=2, color=[0, 102, 204]),
        4: dict(link=('L_Shoulder', 'L_Hip'), id=5, color=[0, 102, 204]),
        7: dict(link=('L_Shoulder', 'L_Elbow'), id=8, color=[0, 102, 204]),
        9: dict(link=('L_Elbow', 'L_Wrist'), id=10, color=[0, 102, 204]),

        # RIGHT (rosso)
        2: dict(link=('R_Knee', 'R_Hip'), id=3, color=[220, 50, 32]),
        5: dict(link=('R_Shoulder', 'R_Hip'), id=6, color=[220, 50, 32]),
        8: dict(link=('R_Shoulder', 'R_Elbow'), id=9, color=[220, 50, 32]),
        10: dict(link=('R_Elbow', 'R_Wrist'), id=11, color=[220, 50, 32]),
    },

    joint_weights=[1.] * 14,
    sigmas = [0.05] * 14,
)
