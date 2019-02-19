"""
visual question answering with "transparent by design" architecture
Contains TBD class to run vqa pipeline
"""

from opencog.type_constructors import FloatValue, types
from opencog.bindlink import execute_atom

from tbd_cog import tbd_helpers
from pattern_matcher_vqa import PatternMatcherVqaPipeline, popAtomspace, pushAtomspace

import sys

class TransparentByDesignVQA(PatternMatcherVqaPipeline):

    def answer_by_programs(self, tdb_net, features, programs):
        """
        Compute answers from image features and programs
        :param tdb_net: tbd.TbDNet
            Object holding neural networks
        :param features: torch.Tensor
            Images features
        :param programs: torch.Tensor
            Programs in numeric form
        :return: List[str]
            answers as strings
        """
        batch_size = features.size(0)
        feat_input_volume = tdb_net.stem(features)
        results = []
        for n in range(batch_size):
            feat_input = feat_input_volume[n:n + 1]
            output = feat_input
            program = []
            for i in reversed(programs.data[n].cpu().numpy()):
                module_type = tdb_net.vocab['program_idx_to_token'][i]
                if module_type == '<NULL>':
                    continue
                program.append(module_type)
            # print ("We're about to call self.run_program with")
            # print ("output = {}".format(output))
            # print ("program  = {}".format(program))
            result = self.run_program(output, program)
            results.append(result)
        print ("we have results: {}".format(results))
        # sys.exit(0)    
        return results

    def argmax(self, answer_set):
        items = []
        key_attention, key_scene, key_shape_attention, key_shape_scene = tbd_helpers.generate_keys(self.atomspace)
        for list_link in answer_set.get_out():
            atoms = list_link.get_out()
            value = -1
            concept = None
            for atom in atoms:
                if atom.name.startswith("Data-"):
                    value_ = tbd_helpers.extract_tensor(atom, key_attention, key_shape_attention)
                    # print ("value_ is placed on device {}".format(value_.device))
                    if str(value_.device).startswith("cuda"):
                        value_ = value_.to("cpu")
                        # print ("value_ is placed on device {}".format(value_.device))
                        # sys.exit(0)
                        value = value_.numpy().sum()
                elif atom.name.startswith("BoundingBox"):
                    continue
                else:
                    concept = atom.name
            assert concept
            assert value != -1
            items.append((value, concept))
        if not items:
            return None
        items.sort(reverse=True)
        return items[0][1]

    def run_program(self, features, program):
        self.atomspace = pushAtomspace(self.atomspace)
        # print ("self.atomspace = {}".format(self.atomspace))
        self._add_scene_atom(features)
        # print ("in tbd_vqa.run_program, we're about to call tbd_helpers.return_prog")
        rev_prog = tuple(reversed(program))
        # print ("reversed program = {}".format(rev_prog))

        eval_link, left, inheritance_set = tbd_helpers.return_prog(commands=rev_prog, atomspace=self.atomspace)
        # print ("We got eval_link = {}".format(eval_link))
        # print ("We got left = {}".format(left))
        # print ("We got inheritance_set = {}".format(inheritance_set))
        
        bind_link = tbd_helpers.build_bind_link(self.atomspace, eval_link, inheritance_set)
        # print ("bind_link = {}".format(bind_link))
        
        # result = bindlink.bindlink(self.atomspace, bind_link)
        
        # TODO: change so that 
        # bindlink.bindlink is no longer used
        result = execute_atom(self.atomspace, bind_link)
        # sys.exit(0)
        # print ("result = {}".format(result))
        answer = self.argmax(result)
        self.atomspace = popAtomspace(self.atomspace)
        return answer

    def _add_scene_atom(self, features):
        _, key_scene, _, key_shape_scene = tbd_helpers.generate_keys(self.atomspace)
        data = FloatValue(list(features.cpu().numpy().flatten()))
        bbox_instance = self.atomspace.add_node(types.ConceptNode, 'BoundingBox1')
        bbox_instance.set_value(key_scene, data)
        bbox_instance.set_value(key_shape_scene, FloatValue(list(features.cpu().numpy().shape)))
        box_concept = self.atomspace.add_node(types.ConceptNode, 'BoundingBox')
        self.atomspace.add_link(types.InheritanceLink, [bbox_instance, box_concept])
