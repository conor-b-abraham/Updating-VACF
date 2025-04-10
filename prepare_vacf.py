import MDAnalysis as mda
from MDAnalysis.analysis.base import AnalysisFromFunction
import numpy as np
import argparse
from tqdm import tqdm
import shutil
import os
import sys

# ------------------------------------------------------------------------------
# INPUT
# ------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='PREPARE ATOM GROUP .npy FILES FOR updating_vacf.jl')
INPUT_FILES = parser.add_argument_group('INPUT')
INPUT_FILES.add_argument('-i', '--internal_water_dir', required=True, help='(REQUIRED) Directory containing internal water selection files (.npy) from find_internal_waters.')
INPUT_FILES.add_argument('-t','--trajectory', required=True, help='(REQUIRED) Trajectory file containing coordinate information (e.g. XTC, TRR, DCD)')
INPUT_FILES.add_argument('-s', '--structure', default='system.pdb', help='(OPTIONAL; default=system.pdb) Structure/topology file containing segids (e.g. PDB)')
OPTIONS = parser.add_argument_group('OPTIONS')
OPTIONS.add_argument('-c', '--external_cutoff', default=12, type=int, help='(OPTIONAL; default=12) Distance for external cutoff (in Angstroms) will separate external waters into those within this number of angstroms of the fibril and those outside 12 angstroms of the fibril')
OPTIONS.add_argument('-n', '--n_internal_groups', default=3, type=int, help='(OPTIONAL; default=3) The number of internal groups to separate the internal water into. For example, 3 will separate the waters inside each of the water channels into thirds (top, middle, and bottom). If an odd number is used, the innermost group will span twice the distance of the other groups.')
args = parser.parse_args()

TRAJ = args.trajectory
STRUC = args.structure

DIR = args.internal_water_dir
if DIR[-1] == '/':
    DIR = DIR[:-1]

internal_sel = [np.load(f'{DIR}/select_H2Oin{i}.npy') for i in range(1,3)] # list of internal water selection arrays
external_sel = np.load(f'{DIR}/select_H2Oout.npy')
OUTDIR = f'{DIR}/vacf_inputs'

if os.path.isdir(OUTDIR):
    shutil.rmtree(OUTDIR)
os.makedirs(OUTDIR)

external_cutoff = args.external_cutoff
n_internal_groups = args.n_internal_groups

print(f"""------------------------------------------------------------------------------
Preparing Atomgroups, Velocities, and Axes for updating_vacf.jl
------------------------------------------------------------------------------
Reading topology from: {STRUC}
Reading trajectory from: {TRAJ}
Reading internal water selections from and writing outputs to: {OUTDIR}
External Cutoff: {external_cutoff} angstroms
Number of Internal Groups: {n_internal_groups}
""")

# ------------------------------------------------------------------------------
# FUNCTIONS
# ------------------------------------------------------------------------------
def segid_array(ag, N_PF):
    '''
    Create (n_layers, n_protofilaments) array of segment IDs

    Parameters
    ----------
    ag : MDAnalysis.atomgroup
        Atom group of the fibril
    N_PF : Int
        The number of protofilaments in the fibril

    Returns
    -------
    segids : numpy.ndarray
        The (n_layers, n_protofilaments) array of segment IDs
    '''
    all_segids = [ag.residues.segids[i] for i in sorted(np.unique(ag.residues.segids, return_index=True)[1])]
    pf_segids = []
    for pf in range(N_PF):
        pf_segids.append(all_segids[pf::N_PF])
    pf_segids = np.array(pf_segids).T
    if N_PF == 1: # Force to be 2d so single protofilament case will behave like multiple protofilament case
        pf_segids = np.atleast_2d(pf_segids)
    return pf_segids

def unit_vector(vector):
    return vector/np.linalg.norm(vector)

def where_internal(u, sidetop, sidebottom, internal_sel, allside):
    # save original positions to make correction after
    original_positions = u.atoms.positions

    # Rotate system to be aligned with water channel
    com1 = sidetop.center_of_mass()
    com2 = sidebottom.center_of_mass()
    Nvec = unit_vector(com1-com2)
    #Nvec = Nvec/np.linalg.norm(Nvec)
    Nref = np.array([1,0,0])
    v = np.cross(Nvec,Nref)
    c = np.dot(Nvec,Nref)
    s = np.linalg.norm(v)
    I = np.identity(3)
    vXStr = '{} {} {}; {} {} {}; {} {} {}'.format(0, -v[2], v[1], v[2], 0, -v[0], -v[1], v[0], 0)
    k = np.matrix(vXStr)
    R = I + k + np.matmul(k,k) * ((1 -c)/(s**2))
    centered_positions = u.atoms.positions-allside.center_of_mass()
    rotated_positions = R.dot(centered_positions.T).T
    u.atoms.positions = rotated_positions

    # create atomgroup for internal waters
    internal_water = u.select_atoms('resname TIP3 and name OH2')[internal_sel]

    # find distances from center of mass of water channel along x axis (direction fibril is now oriented)
    indists = internal_water.positions[:,0]-allside.center_of_mass()[0]

    # reset positions
    u.atoms.positions = original_positions

    return indists, Nvec

# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------
# First, calculate axes and internal water distances
u = mda.Universe(STRUC, TRAJ)

# Correct protein resids (Matching residues in fibril segments need to have the same index)
new_resids = []
for seg in u.select_atoms("protein").segments:
    new_resids += list(range(1, seg.residues.n_residues+1))
new_resids += u.select_atoms("not protein").residues.resids.tolist()
u.residues.resids = np.array(new_resids)

# Create Atom Groups
fibril = u.select_atoms('protein')
allwater_oxygen = u.select_atoms('resname TIP3 and name OH2')
allsegids = segid_array(fibril, 2) # Segid array with 2 protofilaments and not excluding any segids
side1segids = allsegids[:,0]
side2segids = allsegids[:,1]
fibril = fibril.select_atoms('resid 22:50 and name CA')
sidetops = [fibril.select_atoms(f'segid {side1segids[-2]}'), fibril.select_atoms(f'segid {side2segids[-2]}')]
sidebottoms = [fibril.select_atoms(f'segid {side1segids[1]}'), fibril.select_atoms(f'segid {side2segids[1]}')]
allside1 = fibril.select_atoms("segid "+" ".join(side1segids[1:-1]))
allside2 = fibril.select_atoms("segid "+" ".join(side2segids[1:-1]))
fibriltop = sidetops[0]+sidetops[1]
fibrilbottom = sidebottoms[0]+sidebottoms[1]

internal_dists = np.zeros((u.trajectory.n_frames, allwater_oxygen.n_atoms))
channel1_axis = np.zeros((u.trajectory.n_frames, 3))
channel2_axis = np.zeros((u.trajectory.n_frames, 3))
for ts in tqdm(u.trajectory, desc='Calculating Internal Water Distances'):
    # side 1 internal waters distance from center of fibril
    internal_dists[ts.frame, internal_sel[0][ts.frame, :]], channel1_axis[ts.frame, :] = where_internal(u, sidetops[0], sidebottoms[0], internal_sel[0][ts.frame, :], allside1)

    # side 2 internal waters distance from center of fibril
    internal_dists[ts.frame, internal_sel[1][ts.frame, :]], channel2_axis[ts.frame, :] = where_internal(u, sidetops[1], sidebottoms[1], internal_sel[1][ts.frame, :], allside2)
u.trajectory[0] # Safety first: Go back to beginning of trajectory

np.save(f'{OUTDIR}/channel1_axis.npy', channel1_axis)
np.save(f'{OUTDIR}/channel2_axis.npy', channel2_axis)
print(f'    Water channel axes arrays (N_frame x 3) saved to {OUTDIR}/channel1_axis.npy and {OUTDIR}/channel2_axis.npy')
fibril_axis = np.apply_along_axis(unit_vector, 1, (channel1_axis+channel2_axis)/2)
np.save(f'{OUTDIR}/fibril_axis.npy', fibril_axis)
print(f'    Fibril axis array (N_frames x 3) saved to {OUTDIR}/fibril_axis.npy\n')

# find cutoffs for internal waters & find equalities to use
# if using an even number of n_internal_groups then the first minimum and all maximums will include atoms the cutoff distance away (this is messy but probably doesn't matter)
# if an odd number of cutoff groups are used, the cutoff furthest from the center will be inclusive and the innermost group will be inclusive in both directions
print('Creating Internal Water Groups')
maxin = np.ceil(np.max(internal_dists))
minin = np.floor(np.min(internal_dists))
if n_internal_groups % 2 == 0:
    internal_cutoffs = np.linspace(minin, maxin, n_internal_groups+1)
    min_inclusion = [True]*n_internal_groups
    max_inclusion = [False]*n_internal_groups
else:
    temp_internal_cutoffs = np.linspace(minin, maxin, n_internal_groups+2)
    middle_index = int((n_internal_groups+1)/2)
    internal_cutoffs = []
    for i, cutoff in enumerate(temp_internal_cutoffs):
        if i != middle_index:
            internal_cutoffs.append(cutoff)
    internal_cutoffs = np.array(internal_cutoffs)
    min_inclusion, max_inclusion = [], []
    middle_index = int((n_internal_groups-1)/2)
    for i in range(n_internal_groups):
        if i == middle_index:
            min_inclusion.append(True)
            max_inclusion.append(True)
        elif i < middle_index:
            min_inclusion.append(True)
            max_inclusion.append(False)
        elif i > middle_index:
            min_inclusion.append(False)
            max_inclusion.append(True)

# find internal groups
n_frames = internal_dists.shape[0]
n_atoms = internal_dists.shape[1]
internal1_groups = [np.zeros((n_frames, n_atoms), dtype=bool) for i in range(n_internal_groups)]
internal2_groups = [np.zeros((n_frames, n_atoms), dtype=bool) for i in range(n_internal_groups)]
for i, mindis in enumerate(internal_cutoffs[:-1]):
    maxdis = internal_cutoffs[i+1]
    if min_inclusion[i]:
        above_min = np.where(internal_dists >= mindis, True, False)
    else:
        above_min = np.where(internal_dists > mindis, True, False)
    if max_inclusion[i]:
        below_max = np.where(internal_dists <= maxdis, True, False)
    else:
        below_max = np.where(internal_dists < maxdis, True, False)
    # print(above_min.shape, below_max.shape, internal_sel[0].shape)
    internal1_groups[i] = above_min*below_max*internal_sel[0]
    internal2_groups[i] = above_min*below_max*internal_sel[1]

for g in range(n_internal_groups):
    np.save(f'{OUTDIR}/select_in1-{g+1}.npy', internal1_groups[g])
    np.save(f'{OUTDIR}/select_in2-{g+1}.npy', internal2_groups[g])

print("    The following files were saved with groups using the given cutoffs:")
with open(f'{OUTDIR}/about.txt', 'w+') as w:
    w.write(f'INTERNAL (n_groups per side = {n_internal_groups})\n')
    for g in range(n_internal_groups):
        if min_inclusion[g]:
            minop = '<='
        else:
            minop = '<'
        if max_inclusion[g]:
            maxop = '<='
        else:
            maxop = '<'
        minbound = internal_cutoffs[g]
        maxbound = internal_cutoffs[g+1]
        w.write(f'select_in1-{g+1}.npy: {minbound} {minop} d {maxop} {maxbound}\n')
        print(f'        - {OUTDIR}/select_in1-{g+1}.npy: {minbound} {minop} d {maxop} {maxbound}')
print()

# find external groups
exnear_group = np.zeros((n_frames, n_atoms), dtype=bool)
exfar_group = np.zeros((n_frames, n_atoms), dtype=bool)
search_group = u.select_atoms('protein or (resname TIP3 and name OH2)')
exnear = search_group.select_atoms('around 12 protein', updating=True, periodic=False) # Might want to turn on periodic, but I don't think it will work with dodecahedron boxes

for ts in tqdm(u.trajectory, desc='Creating External Water Groups'):
    current_external = allwater_oxygen[external_sel[ts.frame,:]]
    exnear_group[ts.frame, :] = np.isin(allwater_oxygen.indices, (current_external&exnear).indices)
    exfar_group[ts.frame, :] = np.isin(allwater_oxygen.indices, (current_external&(allwater_oxygen-exnear)).indices)

u.trajectory[0] # Safety first: Go back to beginning of trajectory
np.save(f'{OUTDIR}/select_out_near.npy', exnear_group)
np.save(f'{OUTDIR}/select_out_far.npy', exfar_group)

# write an about file
print("    The folling files were saved with groups using the given cutoffs:")
with open(f'{OUTDIR}/about.txt', 'a') as w:
    w.write(f'\nEXTERNAL (cutoff = {external_cutoff})\n')
    w.write(f'select_out_near.npy: d <= {external_cutoff}\n')
    print(f'        - {OUTDIR}/select_out_near.npy: d <= {external_cutoff}')
    w.write(f'select_out_far.npy: d > {external_cutoff}\n')
    print(f'        - {OUTDIR}/select_out_far.npy: d > {external_cutoff}\n\n')

print(f'''
WRITING VELOCITIES FOR SYSTEM FROM
    Topology file: {STRUC}
    Trajectory file: {TRAJ}

CONTAINING
    Number of Atoms: {allwater_oxygen.n_atoms}
    Number of Frames: {u.trajectory.n_frames}
''')

velocities = np.zeros((u.trajectory.n_frames, allwater_oxygen.n_atoms, 3))
for ts in tqdm(u.trajectory, desc=f'Collecting Velocities'):
    velocities[ts.frame, :, :] = allwater_oxygen.velocities

# print('Collecting Velocities:')
# all_velocities = AnalysisFromFunction(lambda atoms: atoms.velocities.copy(), allwater_oxygen).run(verbose=True).results['timeseries']
# print('\033[1A', end='\x1b[2K') # Clear progress bar

np.save(f'{OUTDIR}/velocities.npy', velocities)
print(f'Velocities have been written to {OUTDIR}/velocities.npy')

# # Now reduce each selection array and velocity array for each group
# # External Far
# select_anyfar = np.any(exfar_group, axis=0)
# exfar_group_red = exfar_group[:, select_anyfar]
# exfar_vel = all_velocities[:, select_anyfar, :]
# np.save(f"{OUTDIR}/exfar_velocities.npy", exfar_vel)
# np.save(f'{OUTDIR}/select_out_far_reduced.npy', exfar_group_red)

# # External Near
# select_anynear = np.any(exnear_group, axis=0)
# exnear_group_red = exnear_group[:, select_anynear]
# exnear_vel = all_velocities[:, select_anynear, :]
# np.save(f"{OUTDIR}/exnear_velocities.npy", exnear_vel)
# np.save(f'{OUTDIR}/select_out_near_reduced.npy', exnear_group_red)

# # Internal
# select_anyinternal = []
# for i in internal1_groups:
#     select_anyinternal.append(np.any(i, axis=0))
# for i in internal2_groups:
#     select_anyinternal.append(np.any(i, axis=0))
# select_anyinternal = np.any(np.vstack(select_anyinternal), axis=0)

# for i, select in enumerate(internal1_groups):
#     np.save(f'{OUTDIR}/select_in1-{g+1}_reduced.npy', internal1_groups[g][:, select_anyinternal])
# for i, select in enumerate(internal2_groups):
#     np.save(f'{OUTDIR}/select_in2-{g+1}_reduced.npy', internal2_groups[g][:, select_anyinternal])

# in_vel = all_velocities[:, select_anyinternal, :]
# np.save(f"{OUTDIR}/internal_velocities.npy", in_vel)


