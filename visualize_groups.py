import MDAnalysis as mda
import numpy as np
import sys

# Classes & Functions
class visualization_state:
    '''
    Write a VMD visualization state.
    '''
    def __init__(self, topology_file, trajectory_file, frame, outpdb):
        self.topology_file = topology_file
        self.trajectory_file = trajectory_file
        self.outpdb = outpdb
        self.frame = frame
        self.universe = mda.Universe(self.topology_file, self.trajectory_file)
        self.universe.trajectory[self.frame]
        self.state = ['# Written with mda2vmd',
                     f'mol new {self.outpdb} type pdb first 0 last -1 step 1 filebonds 1 autobonds 1 waitfor all',
                      'mol delrep 0 top\n']

    def add_representation(self, selection, style, color):
        # set color
        default_colors = {'blue':0,'red':1,'gray':2,'grey':2,'orange':3,'yellow':4,'tan':5,'silver':6,
                          'green':7,'white':8,'pink':9,'cyan':10,'purple':11,'lime':12,'mauve':13,'ochre':14,
                          'iceblue':15,'black':16,'yellow2':17,'yellow3':18,'green2':19,'green3':20,'cyan2':21,
                          'cyan3':22,'blue2':23,'blue3':24,'violet':25,'violet2':26,'magenta':27,'magenta2':28,
                          'red2':29,'red3':30,'orange2':31,'orange3':32}
        color = f'ColorID {default_colors[color.lower()]}'

        # set style
        styles = {'licorice':'Licorice 0.300000 12.000000 12.000000',
                  'cpk':'CPK 0.500000 0.300000 12.000000 12.000000',
                  'vdw':'VDW 1.000000 12.000000',
                  'newcartoon':'NewCartoon 0.300000 10.000000 4.100000 0'}
        style = styles[style.lower()]

        # create atomgroup from selection
        indices = self.universe.select_atoms(selection).atoms.indices.tolist()
        indices.sort()
        if indices == list(range(indices[0], indices[-1]+1)):
            selection_command = '{'+f'index {indices[0]} to {indices[-1]}'+'}'
        else:
            selection_command = '{'+f'index {" ".join([str(i) for i in indices])}'+'}'

        # add to state
        self.state.append(f'mol representation {style}')
        self.state.append(f'mol color {color}')
        self.state.append(f'mol selection {selection_command}')
        self.state.append('mol material AOShiny')
        self.state.append('mol addrep top\n')

    def write(self, filename):
        self.state.append(f'mol rename top {self.outpdb}')
        with open(filename, 'w+') as io:
            for line in self.state:
                io.write(f'{line}\n')
        self.universe.atoms.write(self.outpdb)

# Main
if len(sys.argv) != 3:
    print("""USAGE: python visualize_groups.py [TRAJ] [TOP]
        where,
            TRAJ = Trajectory File
            TOP = Topology File""")
    sys.exit()

TRAJ = sys.argv[1]
TOP = sys.argv[2]
frame = 5000
u = mda.Universe(TOP, TRAJ)
u.trajectory[5000]
allwater = u.select_atoms("resname TIP3 and name OH2")
v = visualization_state(TOP, TRAJ, 5000, "visgroups.pdb")
v.add_representation('protein', 'NewCartoon', 'gray')
for i, (files, color) in enumerate(zip([["select_out_far.npy"], ["select_out_near.npy"], ["select_in1-1.npy", "select_in1-3.npy", "select_in2-1.npy", "select_in2-3.npy"], ["select_in1-2.npy", "select_in2-2.npy"]], ["mauve", "green3", "blue2", "orange3"])):
    for file in files:
        print(file, color)
        sel = np.load(file)[5000,:]
        if i < 3:
            addon = "prop x > 90 and prop x < 130 and "
        else:
            addon = ""
        v.add_representation(addon+'resname TIP3 and index '+' '.join([str(i) for i in allwater.indices[sel]]), 'vdw', color)
    v.write("visgroups.vmd")