from ._initial_preparation_parent import InitialPreparationParent


class InitialPreparationTypedWrdsImgesDatasets(InitialPreparationParent):
    def __init__(self, dataset_directory):
        super().__init__(dataset_directory)

    def _list_data_names_and_labels(self):
        pass

    def generate_data_labels(self):
        pass
