import os
import numpy as np
import util_shared
import matplotlib.pyplot as plt
import scipy.io

arm_filename = 'arm_touchread_sensitive_ikea_6sensorsnewimproved_pi8_15cm_iteration_%d_movement_%s.pkl'
leg_filename = 'leg_touchread_sensitive_ikea_6sensorsnewimproved_pi8_15cm_iteration_%d_movement_%s.pkl'
directory = 'trainingdata'

def load_data(participants=range(2, 14), arm=True, limb_segment='wrist', trajectory=None):
    # Load collected capacitance data and ground truth pose
    seg = {'wrist': 0, 'forearm': 1, 'upperarm': 2, 'ankle': 0, 'shin': 1, 'knee': 2}[limb_segment]
    files = [os.path.join('participant_%d' % p, (arm_filename if arm else leg_filename) % (seg, wildcard)) for p in participants for wildcard in (['?', '??', '???'] if trajectory is None else [str(trajectory)])]
    capacitance, pos_orient, times = util_shared.load_data_from_file(files=files, directory=directory)
    print(np.shape(capacitance), np.shape(pos_orient))
    pos_y, pos_z, angle_y, angle_z = pos_orient[:, 0], pos_orient[:, 1], pos_orient[:, 2], pos_orient[:, 3]
    # return capacitance, pos_y, pos_z, angle_y, angle_z
    return capacitance, pos_orient

#load_data(participants=[2], arm=True, limb_segment='wrist')

for k in range(2,12):
    for i in range(1,50):
        c, po = load_data(participants=[k], arm=True, limb_segment='shin', trajectory=i)
        name = 'participant_%d_shin_traj_%d.mat' % (k,i)
        scipy.io.savemat(name, dict(x=po, y=c))

#all_capacitance = []
#all_pos_orient = []
#for i, limb_segment in enumerate(['wrist']): #, 'forearm', 'upperarm', 'ankle', 'shin', 'knee']):
#    c, po = load_data(participants=[2], arm=True, limb_segment=limb_segment)
#    all_capacitance.append(c)
#    all_pos_orient.append(po)
#all_capacitance = np.concatenate(all_capacitance, axis=0)
#all_pos_orient = np.concatenate(all_pos_orient, axis=0)
#print('All capacitance shape:', all_capacitance.shape, 'All pos_orient shape:', all_pos_orient.shape)

#scipy.io.savemat('participant_2_forearm_traj_51.mat', dict(x=po,y=c))

#plt.plot(c)
#plt.show()
#plt.legend()