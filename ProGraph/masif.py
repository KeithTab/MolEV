## 1.BEFORE YOU RUN THIS SCRIPTS ,YOU'D BETTER DEFINE YOUR OWN PATH AS FOLLOWS
## 2.RUN THIS SCRIPTS BETTER UNDER THE LINUX SYSTEM
## 3.REFERENCE: https://github.com/casperg92/dMasif

import os
import glob
pred_dir = '/home/szk/content/pdbs'
isExist = os.path.exists(pred_dir)
if not isExist:
  os.makedirs(pred_dir)

os.chdir('/home/szk/content')
target_pdb = "/home/szk/content/pdbs/1aki.pdb" 
target_name = target_pdb.split('/')
target_name = target_name[-1].split('.')

if target_name[-1] == 'pdb':
  target_name = target_name[0]
else:
  print('Please upload a valid .pdb file!')

chain_name = 'A'
chains = [chain_name]

model_resolution = '0.7 Angstrom' 
patch_radius = '9 Angstrom' 


if patch_radius == '9 Angstrom':
  if model_resolution == '1 Angstrom':
    model_path = '/home/szk/content/MaSIF_colab/models/dMaSIF_site_3layer_16dims_9A_100sup_epoch64'
    resolution = 1.0
    radius = 9
    sup_sampling = 100
  else:
    model_path = '/home/szk/content/MaSIF_colab/models/dMaSIF_site_3layer_16dims_9A_0.7res_150sup_epoch85'
    resolution = 0.7
    radius = 9
    supsampling = 150

elif patch_radius == '12 Angstrom':
  if model_resolution == '1 Angstrom':
    model_path = '/home/szk/content/MaSIF_colab/models/dMaSIF_site_3layer_16dims_12A_100sup_epoch71'
    resolution = 1.0
    radius = 12
    supsampling = 100
  else:
    model_path = '/home/szk/content/MaSIF_colab/models/dMaSIF_site_3layer_16dims_12A_0.7res_150sup_epoch59'
    resolution = 0.7
    radius = 12
    supsampling = 100

chains_dir = '/home/szk/content/chains'
isExist = os.path.exists(chains_dir)
if not isExist:
  os.makedirs(chains_dir)
else:
  files = glob.glob(chains_dir + '/*')
  for f in files:
    os.remove(f)

npy_dir = '/home/szk/content/npys'
isExist = os.path.exists(npy_dir)
if not isExist:
  os.makedirs(npy_dir)
else:
  files = glob.glob(npy_dir + '/*')
  for f in files:
    os.remove(f)

pred_dir = '/home/szk/content/preds'
isExist = os.path.exists(pred_dir)
if not isExist:
  os.makedirs(pred_dir)
else:
  files = glob.glob(pred_dir + '/*')
  for f in files:
    os.remove(f)

import sys
sys.path.append("/home/szk/content/MaSIF_colab") 
sys.path.append("/home/szk/content/MaSIF_colab/data_preprocessing") 

import numpy as np
import pykeops
import torch
from Bio.PDB import *
from data_preprocessing.download_pdb import convert_to_npy
from torch_geometric.data import DataLoader
from torch_geometric.transforms import Compose
import argparse
import shutil

from data import ProteinPairsSurfaces, PairData, CenterPairAtoms, load_protein_pair
from data import RandomRotationPairAtoms, NormalizeChemFeatures, iface_valid_filter
from model import dMaSIF
from data_iteration import iterate
from helper import *

import nglview as ng
from pdbparser.pdbparser import pdbparser


def generate_descr(model_path, output_path, pdb_file, npy_directory, radius, resolution,supsampling):

    """Generat descriptors for a MaSIF site model"""
    parser = argparse.ArgumentParser(description="Network parameters")
    parser.add_argument("--experiment_name", type=str, default=model_path)
    parser.add_argument("--use_mesh", type=bool, default=False)
    parser.add_argument("--embedding_layer",type=str,default="dMaSIF")
    parser.add_argument("--curvature_scales",type=list,default=[1.0, 2.0, 3.0, 5.0, 10.0])
    parser.add_argument("--resolution",type=float,default=resolution)
    parser.add_argument("--distance",type=float,default=1.05)
    parser.add_argument("--variance",type=float,default=0.1)
    parser.add_argument("--sup_sampling", type=int, default=supsampling)
    parser.add_argument("--atom_dims",type=int,default=6)
    parser.add_argument("--emb_dims",type=int,default=16)
    parser.add_argument("--in_channels",type=int,default=16)
    parser.add_argument("--orientation_units",type=int,default=16)
    parser.add_argument("--unet_hidden_channels",type=int,default=8)
    parser.add_argument("--post_units",type=int,default=8)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--radius", type=float, default=radius)
    parser.add_argument("--k",type=int,default=40)
    parser.add_argument("--dropout",type=float,default=0.0)
    parser.add_argument("--site", type=bool, default=True) 
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--search",type=bool,default=False)
    parser.add_argument("--single_pdb",type=str,default=pdb_file)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random_rotation",type=bool,default=False)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--single_protein",type=bool,default=True) 
    parser.add_argument("--no_chem", type=bool, default=False)
    parser.add_argument("--no_geom", type=bool, default=False)
    
    args = parser.parse_args("")

    model_path = args.experiment_name
    save_predictions_path = Path(output_path)
    
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    transformations = (
        Compose([NormalizeChemFeatures(), CenterPairAtoms(), RandomRotationPairAtoms()])
        if args.random_rotation
        else Compose([NormalizeChemFeatures()])
    )
    
    if args.single_pdb != "":
        single_data_dir = Path(npy_directory)
        test_dataset = [load_protein_pair(args.single_pdb, single_data_dir, single_pdb=True)]
        test_pdb_ids = [args.single_pdb]

    batch_vars = ["xyz_p1", "xyz_p2", "atom_coords_p1", "atom_coords_p2"]
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, follow_batch=batch_vars
    )

    net = dMaSIF(args)
    net.load_state_dict(torch.load(model_path, map_location=args.device)["model_state_dict"])
    net = net.to(args.device)

    info = iterate(
        net,
        test_loader,
        None,
        args,
        test=True,
        save_path=save_predictions_path,
        pdb_ids=test_pdb_ids,
    )
    return info

def show_pointcloud(main_pdb, coord_file, emb_file):

  b_factor = []
  for emb in emb_file:
      b_factor.append(emb[-2])
  
  records = []

  for i in range(len(coord_file)):
      points = coord_file[i]
      x_coord = points[0]
      y_coord = points[1]
      z_coord = points[2]

      records.append( { "record_name"       : 'ATOM',
                    "serial_number"     : len(records)+1,
                    "atom_name"         : 'H',
                    "location_indicator": '',
                    "residue_name"      : 'XYZ',
                    "chain_identifier"  : '',
                    "sequence_number"   : len(records)+1,
                    "code_of_insertion" : '',
                    "coordinates_x"     : x_coord,
                    "coordinates_y"     : y_coord,
                    "coordinates_z"     : z_coord,
                    "occupancy"         : 1.0,
                    "temperature_factor": b_factor[i]*100,
                    "segment_identifier": '',
                    "element_symbol"    : 'H',
                    "charge"            : '',
                    } )
    
  pdb = pdbparser()
  pdb.records = records

  pdb.export_pdb("pointcloud.pdb")

  coordPDB = "pointcloud.pdb"
  
  view = ng.NGLWidget()
  view.add_component(ng.FileStructure(os.path.join("/home/szk/content", coordPDB)), defaultRepresentation=False)

  view.add_representation('point', 
                          useTexture = 1,
                          pointSize = 2,
                          colorScheme = "bfactor",
                          colorDomain = [100.0, 0.0], 
                          colorScale = 'rwb',
                          selection='_H')

  view.add_component(ng.FileStructure(os.path.join("/home/szk/content", main_pdb)))
  view.background = 'black'
  return view

def show_structure(main_pdb):

  view = ng.NGLWidget()

  view.add_component(ng.FileStructure(main_pdb), defaultRepresentation=False)
  view.add_representation("cartoon", colorScheme = "bfactor", colorScale = 'rwb', colorDomain = [100.0, 0.0])
  view.add_representation("ball+stick", colorScheme = "bfactor", colorScale = 'rwb', colorDomain = [100.0, 0.0])
  view.background = 'black'
  return view

tmp_pdb = '/home/szk/content/pdbs/tmp_1.pdb'
shutil.copyfile(target_pdb, tmp_pdb)

os.system('reduce -Trim -Quiet /home/szk/content/pdbs/tmp_1.pdb > /home/szk/content/pdbs/tmp_2.pdb')

os.system('reduce -HIS -Quiet /home/szk/content/pdbs/tmp_2.pdb > /home/szk/content/pdbs/tmp_3.pdb')

tmp_pdb = '/home/szk/content/pdbs/tmp_3.pdb'
shutil.copyfile(tmp_pdb, target_pdb)

convert_to_npy(target_pdb, chains_dir, npy_dir, chains)

pdb_name = "{n}_{c}_{c}".format(n= target_name, c=chain_name)
info = generate_descr(model_path, pred_dir, pdb_name, npy_dir, radius, resolution, supsampling)

list_hotspot_residues = False 

from Bio.PDB.PDBParser import PDBParser
from scipy.spatial.distance import cdist

parser=PDBParser(PERMISSIVE=1)
structure=parser.get_structure("structure", target_pdb)

coord = np.load("/home/szk/content/preds/{n}_{c}_predcoords.npy".format(n= target_name, c=chain_name))
embedding = np.load("/home/szk/content/preds/{n}_{c}_predfeatures_emb1.npy".format(n= target_name, c=chain_name))
atom_coords = np.stack([atom.get_coord() for atom in structure.get_atoms()])

b_factor = embedding[:, -2]

dists = cdist(atom_coords, coord)
nn_ind = np.argmin(dists, axis=1)
dists = dists[np.arange(len(dists)), nn_ind]
atom_b_factor = b_factor[nn_ind]
dist_thresh = 2.0
atom_b_factor[dists > dist_thresh] = 0.0

for i, atom in enumerate(structure.get_atoms()):
    atom.set_bfactor(atom_b_factor[i] * 100)

pred_dir = '/home/szk/content/output'
os.makedirs(pred_dir, exist_ok=True)

io = PDBIO()
io.set_structure(structure)
io.save("/home/szk/content/output/per_atom_binding.pdb")

atom_residues = np.array([atom.get_parent().id[1] for atom in structure.get_atoms()])

hotspot_res = {}
for residue in structure.get_residues():
    res_id = residue.id[1]
    res_b_factor = np.max(atom_b_factor[atom_residues == res_id])
    hotspot_res[res_id] = res_b_factor
    for atom in residue.get_atoms():
        atom.set_bfactor(res_b_factor * 100)

io = PDBIO()
io.set_structure(structure)
io.save("/home/szk/content/output/per_resi_binding.pdb")

if list_hotspot_residues:
  print('Sorted on residue contribution (high to low')
  for w in sorted(hotspot_res, key=hotspot_res.get, reverse=True):
    print(w, hotspot_res[w])

os.system('pwd')

plot_structure = 'Pointcloud'
if plot_structure == 'Pointcloud':
  view = show_pointcloud(target_pdb, coord, embedding)
elif plot_structure == "Residues":
  view = show_structure('/content/output/per_resi_binding.pdb')
elif plot_structure == "Atoms":
  view = show_structure('/content/output/per_atom_binding.pdb')