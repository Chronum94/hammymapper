from pprint import pprint
from hammymapper.hblockmapper import make_mapper_from_elements
from hammymapper.utilities import get_pair_mapping_spec, get_element_mapping_spec

hmapper = make_mapper_from_elements({6: [0, 1, 2], 7: [0, 1, 2]})
pprint(hmapper)
# pprint(get_pair_mapping_spec([0, 1, 2], [0, 1, 2], False))
# pprint(get_element_mapping_spec([0, 1]))