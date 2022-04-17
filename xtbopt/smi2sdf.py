import argparse
import multiprocessing as mp
import rdkit
import sys
from contextlib import closing
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign
#导入
def RobustSmilesMolSupplier(filename):
    with open(filename) as f:
        for line in f:
            words = line.split()
            smile = words[0]
            name = words[1]
            yield name, Chem.MolFromSmiles(smile)


def how_many_conformers(mol):
    nb_rot_bonds = AllChem.CalcNumRotatableBonds(mol)
    if nb_rot_bonds <= 7:
        return 50
    elif nb_rot_bonds <= 12:
        return 200
    return 300 


def rmsd_filter(mol, ref_conf, conf_energies, threshold):

    mol_noH = Chem.RemoveHs(mol)
    ref_conf_id = ref_conf.GetId()
    res = []
    for e, curr_conf in conf_energies:
        curr_conf_id = curr_conf.GetId()
        rms = AllChem.GetConformerRMS(mol_noH, ref_conf_id, curr_conf_id)
        if rms > threshold:
            res.append((e, curr_conf))
    return res

def process_one(name, mol, n_confs):
    n = how_many_conformers(mol)
    print("init pool size for %s: %d" % (name, n), file = sys.stderr)
    mol_H = Chem.AddHs(mol)
    res = Chem.Mol(mol_H)
    res.RemoveAllConformers()
    print("generating starting conformers ...", file = sys.stderr)
    conf_energies = []
    print("FF minimization ...", file = sys.stderr)
    for cid in AllChem.EmbedMultipleConfs(mol_H, n):
        ff = AllChem.UFFGetMoleculeForceField(mol_H, confId = cid)
        ff.Minimize()
        energy = ff.CalcEnergy()
        conformer = mol_H.GetConformer(cid)
        conf_energies.append((energy, conformer))

    conf_energies = sorted(conf_energies, key = lambda x: x[0])

    kept = 0
    print("RMSD pruning ...", file = sys.stderr)
    while kept < n_confs and len(conf_energies) > 0:
        (e, conf) = conf_energies.pop(0)
        kept += 1
        cid = res.AddConformer(conf, assignId = True)
        if cid != 0:
            rdMolAlign.AlignMol(res, res, prbCid = cid, refCid = 0)
        conf_energies = rmsd_filter(mol_H, conf, conf_energies, rmsd_threshold)
    print("kept %d confs for %s" % (kept, name), file = sys.stderr)
    name_res = (name, res)
    return name_res

def worker_process(jobs_q, results_q, n_confs):
    for name, mol in iter(jobs_q.get, 'STOP'):
        confs = process_one(name, mol, n_confs)
        results_q.put(confs)
    results_q.put('STOP')

def write_out_confs(rename, name_confs, writer):
    name, confs = name_confs
    for c in confs.GetConformers():
        cid = c.GetId()
        if rename:
            name_cid = "%s_%03d" % (name, cid)
            confs.SetProp("_Name", name_cid)
        else:
            confs.SetProp("_Name", name)
        writer.write(confs, confId = cid)

def multiplexer_process(rename, results_q, output_sdf, nb_workers):
    with closing(Chem.SDWriter(output_sdf)) as writer:
        for i in range(nb_workers):
            for confs in iter(results_q.get, 'STOP'):
                write_out_confs(rename, confs, writer)

#argparse替代人工修改
if __name__ == '__main__':
    rmsd_threshold = 0.35 # A‘
    parser = argparse.ArgumentParser(
        description = "generate diverse low energy 3D conformers; \
        up to [n_confs] per molecule from the input file")
    parser.add_argument("-n", metavar = "n_confs", type = int, default = 1,
                        dest = "n_confs",
                        help = "#conformers per molecule (default: 1)")
    parser.add_argument("-j", metavar="n_procs", type = int, default = 1,
                        dest = "n_procs",
                        help = "max number of parallel jobs (default: 1)")
    parser.add_argument("-i", metavar = "input_smi", dest = "input_smi")
    parser.add_argument("-o", metavar = "output_sdf", dest = "output_sdf")
    parser.add_argument('--rename', dest='rename', action='store_true',
                        help = "append conformer id to molecule name \
                        (default=False")
    parser.set_defaults(rename=False)

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()
    n_confs = args.n_confs
    input_smi = args.input_smi
    output_sdf = args.output_sdf
    n_procs = args.n_procs # 并行运算
    rename = args.rename
    if n_procs > 1:

        jobs_queue = mp.Queue()
        results_queue = mp.Queue()
        for i in range(n_procs):
            mp.Process(target = worker_process,
                       args = (jobs_queue, results_queue, n_confs)).start()
        mp.Process(target = multiplexer_process,
                   args = (rename, results_queue, output_sdf, n_procs)).start()

        for name, mol in RobustSmilesMolSupplier(input_smi):
            if mol is None:
                continue
            jobs_queue.put((name, mol))

        for i in range(n_procs):
            jobs_queue.put('STOP')
    else:

        with closing(Chem.SDWriter(output_sdf)) as writer:
            for name, mol in RobustSmilesMolSupplier(input_smi):
                if mol is None:
                    continue
                conformers = process_one(name, mol, n_confs)
                write_out_confs(rename, conformers, writer)