import numpy as np


class MolBuilder:
    def __init__(self, molecule):

        self.mol = molecule
        self.bound = False

    def initialize_binders(
        self, list_of_molecules, molecule_atom_numbers, binder_atoms_numbers
    ):

        assert len(list_of_molecules) == len(molecule_atom_numbers)
        assert len(list_of_molecules) == len(binder_atoms_numbers)
        self.binders = list_of_molecules
        self.molecule_atom_numbers = molecule_atom_numbers
        self.binder_atoms_numbers = binder_atoms_numbers

    def setup_recursive_binders(self, binder_objects):

        self.binder_objects = binder_objects

    def sample_binder(self):

        if not self.bound:
            (
                chosen_binders,
                chosen_mol_nums,
                chosen_binder_atoms,
                chosen_binder_objects,
            ) = ([], [], [], [])
            self.bound = True
            for unique_node_atom in np.unique(np.array(self.molecule_atom_numbers).T):
                possible_linker_indices = np.arange(len(self.molecule_atom_numbers))[
                    np.where(
                        np.array(self.molecule_atom_numbers).T == unique_node_atom,
                        True,
                        False,
                    )
                ]
                chosen_index = np.random.choice(possible_linker_indices)

                if self.binder_atoms_numbers[chosen_index] in np.unique(
                    self.binder_objects[chosen_index].molecule_atom_numbers
                ):
                    chosen_binders.append(self.binders.pop(chosen_index))
                    chosen_mol_nums.append(self.molecule_atom_numbers.pop(chosen_index))

                    chosen_binder_atom_indice = self.binder_atoms_numbers.pop(
                        chosen_index
                    )
                    chosen_binder_atoms.append(chosen_binder_atom_indice)
                    chosen_binder_object = self.binder_objects.pop(chosen_index)
                    chosen_binder_object.remove_bindable_atom(chosen_binder_atom_indice)

                    chosen_binder_objects.append(chosen_binder_object)

            return list(
                np.array(
                    [
                        chosen_binders,
                        chosen_mol_nums,
                        chosen_binder_atoms,
                        chosen_binder_objects,
                    ]
                ).T
            )
        else:
            return None

    def remove_bindable_atom(self, atom_indice):

        assert atom_indice in self.molecule_atom_numbers

        indices_to_keep = np.arange(len(self.binders))[
            np.where(np.array(self.molecule_atom_numbers) == atom_indice, False, True)
        ]

        self.binders = list(np.array(self.binders)[indices_to_keep])
        self.molecule_atom_numbers = list(
            np.array(self.molecule_atom_numbers)[indices_to_keep]
        )
        self.binder_atoms_numbers = list(
            np.array(self.binder_atoms_numbers)[indices_to_keep]
        )
        self.binder_objects = list(np.array(self.binder_objects)[indices_to_keep])
