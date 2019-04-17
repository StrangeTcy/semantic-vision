from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F  


from tbd.utils.clevr import load_vocab, ClevrDataLoaderH5

import os
import sys # only used for debugging 

# import numpy as np

from tqdm import tqdm #nice progress bars

from opencog.type_constructors import *
from opencog.utilities import initialize_opencog, finalize_opencog

from opencog.scheme_wrapper import scheme_eval_as, scheme_eval

from opencog.bindlink import execute_atom

# ...now we should use the new & improved cogModules from module
from module import CogModule, execute, get_value, evaluate, InputModule, EVALMODE  

from module import CogModule, CogModel, get_value
from module import InputModule, set_value



clevr_answer_map_int_str = {23: 'metal', 
                            27: 'rubber', 
                            22: 'large', 
                            29: 'sphere', 
                            28: 'small', 
                            19: 'cylinder', 
                            17: 'cube',
                            30:'yellow', 
                            26:'red', 
                            25: 'purple', 
                            21:'green', 
                            20:'gray', 
                            18:'cyan', 
                            16:'brown', 
                            15:'blue'}

clevr_answers_map_str_int = {v: k for (k, v) in clevr_answer_map_int_str.items()}



class Stem(CogModule):
    """
    The stem takes features from ResNet
    (or another feature extractor) 
    and projects down to
    a lower-dimensional space 
    for sending through the TbD-net
    """
    def __init__(self, atom, device, feature_dim, module_dim, loaded_state_dict):
        super().__init__(atom)
        self.stem = nn.Sequential(nn.Conv2d(feature_dim[0], module_dim, kernel_size=3, padding=1),
                              nn.ReLU(),
                              nn.Conv2d(module_dim, module_dim, kernel_size=3, padding=1),
                              nn.ReLU()
                             )
        self.stem = self.stem.to(device)
       
        self_stem_part = [key for key, value in self.stem.state_dict().items() ] #if "stem" in key
        if not loaded_state_dict == None:
            for _, (key, value) in enumerate(self.stem.state_dict().items()):
               value = loaded_state_dict[_][1]
         

    def forward(self, x):
        print ("stem got x = {}".format(x))
        return self.stem(x) 




class AndModule(CogModule):
    """ A neural module that 
    (basically) performs a logical and.

    Extended Summary
    ---------------- 
    An :class:`AndModule` is a neural module that takes two input attention masks and (basically)
    performs a set intersection. This would be used in a question like "What color is the cube to
    the left of the sphere and right of the yellow cylinder?" After localizing the regions left of
    the sphere and right of the yellow cylinder, an :class:`AndModule` would be used to find the
    intersection of the two. Its output would then go into an :class:`AttentionModule` that finds
    cubes.
    """
    
    # def __init__(self, atom):
    #     super().__init__(atom)
    

    def forward(self, attn1, attn2):
        out = torch.min(attn1, attn2)
        return out


class OrModule(CogModule):
    """ A neural module that (basically) performs a logical or.

    Extended Summary
    ----------------
    An :class:`OrModule` is a neural module that takes two input attention masks and (basically)
    performs a set union. This would be used in a question like "How many cubes are left of the
    brown sphere or right of the cylinder?" After localizing the regions left of the brown sphere
    and right of the cylinder, an :class:`OrModule` would be used to find the union of the two. Its
    output would then go into an :class:`AttentionModule` that finds cubes.
    """

    # def __init__(self, atom):
    #     super().__init__(atom)
      

    def forward(self, attn1, attn2):
        out = torch.max(attn1, attn2)
        return out



class AttentionModule(CogModule):
    """ A neural module 
    that takes a feature map and attention, 
    attends to the features, 
    and produces an attention.

    Extended Summary
    ----------------
    An :class:`AttentionModule` takes input features and an attention and produces an attention. It
    multiplicatively combines its input feature map and attention to attend to the relevant region
    of the feature map. It then processes the attended features via a series of convolutions and
    produces an attention mask highlighting the objects that possess the attribute the module is
    looking for.

    For example, an :class:`AttentionModule` may be tasked with finding cubes. Given an input
    attention of all ones, it will highlight all the cubes in the provided input features. Given an
    attention mask highlighting all the red objects, it will produce an attention mask highlighting
    all the red cubes.

    Attributes
    ----------
    dim : int
        The number of channels of each convolutional filter.
    """
   
    def __init__(self, atom, dim):
        super().__init__(atom)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1) 
        self.conv3 = nn.Conv2d(dim, 1, kernel_size=1, padding=0)
        torch.nn.init.kaiming_normal_(self.conv1.weight)
        torch.nn.init.kaiming_normal_(self.conv2.weight)
        torch.nn.init.kaiming_normal_(self.conv3.weight)
        self.dim = dim
            

    def forward(self, feats, attn):
        
        feats = feats.to(device)
        attn = attn.to(device)
        attended_feats = torch.mul(feats, attn.repeat(1, self.dim, 1, 1))
        out = F.relu(self.conv1(attended_feats))
        out = F.relu(self.conv2(out))
        out = torch.sigmoid(self.conv3(out))
        return out


class QueryModule(CogModule):
    """ A neural module 
    that takes as input a feature map and an attention 
    and produces a feature map as output.

    Extended Summary
    ----------------
    A :class:`QueryModule` takes a feature map and an attention mask as input. It attends to the
    feature map via an elementwise multiplication with the attention mask, then processes this
    attended feature map via a series of convolutions to extract relevant information.

    For example, a :class:`QueryModule` tasked with determining the color of objects would output a
    feature map encoding what color the attended object is. A module intended to count would output
    a feature map encoding the number of attended objects in the scene.

    Attributes
    ----------
    dim : int
        The number of channels of each convolutional filter.
    """
    def __init__(self, atom, dim):
        super().__init__(atom)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        torch.nn.init.kaiming_normal_(self.conv1.weight)
        torch.nn.init.kaiming_normal_(self.conv2.weight)
        self.dim = dim
            

    def forward(self, feats, attn):
        attended_feats = torch.mul(feats, attn.repeat(1, self.dim, 1, 1))
        out = F.relu(self.conv1(attended_feats))
        out = F.relu(self.conv2(out))
        return out


class RelateModule(CogModule):
    """ A neural module that takes as input a feature map and an attention and produces an attention
    as output.

    Extended Summary
    ----------------
    A :class:`RelateModule` takes input features and an attention and produces an attention. It
    multiplicatively combines the attention and the features to attend to a relevant region, then
    uses a series of dilated convolutional filters to indicate a spatial relationship to the input
    attended region.

    Attributes
    ----------
    dim : int
        The number of channels of each convolutional filter.
    """

    def __init__(self, atom, dim):
        super().__init__(atom)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, dilation=1)  # receptive field 3
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=2, dilation=2)  # 7
        self.conv3 = nn.Conv2d(dim, dim, kernel_size=3, padding=4, dilation=4)  # 15
        self.conv4 = nn.Conv2d(dim, dim, kernel_size=3, padding=8, dilation=8)  # 31 -- full image
        self.conv5 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, dilation=1)
        self.conv6 = nn.Conv2d(dim, 1, kernel_size=1, padding=0)
        torch.nn.init.kaiming_normal_(self.conv1.weight)
        torch.nn.init.kaiming_normal_(self.conv2.weight)
        torch.nn.init.kaiming_normal_(self.conv3.weight)
        torch.nn.init.kaiming_normal_(self.conv4.weight)
        torch.nn.init.kaiming_normal_(self.conv5.weight)
        torch.nn.init.kaiming_normal_(self.conv6.weight)
        self.dim = dim
         

    def forward(self, feats, attn):
        feats = feats.to(device)
        attn = attn.to(device)
        feats = torch.mul(feats, attn.repeat(1, self.dim, 1, 1))
        out = F.relu(self.conv1(feats))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = F.relu(self.conv5(out))
        out = torch.sigmoid(self.conv6(out))
        return out


class SameModule(CogModule):
    """ A neural module that takes as input a feature map and an attention and produces an attention
    as output.

    Extended Summary
    ----------------
    A :class:`SameModule` takes input features and an attention and produces an attention. It
    determines the index of the maximally-attended object, extracts the feature vector at that
    spatial location, then performs a cross-correlation at each spatial location to determine which
    other regions have this same property. This correlated feature map then goes through a
    convolutional block whose output is an attention mask.

    As an example, this module can be used with the CLEVR dataset to perform the `same_shape`
    operation, which will highlight every region of an image that shares the same shape as an object
    of interest (excluding the original object).

    Attributes
    ----------
    dim : int
        The number of channels in the input feature map.
    """

    def __init__(self, atom, dim):
        super().__init__(atom)
        self.conv = nn.Conv2d(dim+1, 1, kernel_size=1)
        torch.nn.init.kaiming_normal_(self.conv.weight)
        self.dim = dim
         

    def forward(self, feats, attn):
        feats = feats.to(device)
        attn = attn.to(device)
        size = attn.size()[2]
        the_max, the_idx = F.max_pool2d(attn, size, return_indices=True)
        attended_feats = feats.index_select(2, the_idx[0, 0, 0, 0] / size)
        attended_feats = attended_feats.index_select(3, the_idx[0, 0, 0, 0] % size)
        x = torch.mul(feats, attended_feats.repeat(1, 1, size, size))
        x = torch.cat([x, attn], dim=1)
        out = torch.sigmoid(self.conv(x))
        return out


class ComparisonModule(CogModule):
    """ A neural module that 
    takes as input two feature maps 
    and produces a feature map as output.

    Extended Summary
    ----------------
    A :class:`ComparisonModule` takes two feature maps as input and concatenates these. It then
    processes the concatenated features and produces a feature map encoding whether the two input
    feature maps encode the same property.

    This block is useful in making integer comparisons, for example to answer the question, ``Are
    there more red things than small spheres?'' It can also be used to determine whether some
    relationship holds of two objects (e.g. they are the same shape, size, color, or material).

    Attributes
    ----------
    dim : int
        The number of channels of each convolutional filter.
    """
    def __init__(self, atom, dim):
        super().__init__(atom)
        self.projection = nn.Conv2d(2*dim, dim, kernel_size=1, padding=0)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        torch.nn.init.kaiming_normal_(self.conv1.weight)
        torch.nn.init.kaiming_normal_(self.conv2.weight)
        

    def forward(self, in1, in2):
        out = torch.cat([in1, in2], 1)
        out = F.relu(self.projection(out))
        out = F.relu(self.conv1(out))
        out = F.relu(self.conv2(out))
        return out






def main():

    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    import re
    import uuid

    relate_reg = re.compile('[a-z]+\[([a-z]+)\]')
    filter_reg = re.compile('[a-z]+_([a-z]+)\[([a-z]+)\]')
    filter_reg.match('filter_shape[sphere]').groups()
    relate_reg.match('relate[front]').groups()

    feature_dim = (1024, 14, 14)
    module_dim = 128

    module_rows, module_cols = feature_dim[1], feature_dim[2]
    ones = torch.ones(1, 1, module_rows, module_cols)
    ones_var = ones.to(device)


    tbd_net_checkpoint = Path('/media/enoch/0645F864324D53D4/neural_stuff/tbd/')
    files = [x for x in tbd_net_checkpoint.iterdir() if x.is_file()]
    
    tbd_net_checkpoint = files[0]
    vocab = load_vocab(Path('/media/enoch/0645F864324D53D4/neural_stuff/tbd/data/vocab.json'))  #'/mnt/fileserver/shared/models/tbd-nets-models/data/vocab.json'
    
    # we have to initialize atomspace here, or Node creation won't work
    atomspace = AtomSpace()
    initialize_opencog(atomspace)

       

    checkpoint = tbd_net_checkpoint
    load_obj = torch.load(checkpoint)

    stem_part = [(key,value) for key, value in load_obj.items() if "stem" in key]

    stem = Stem(ConceptNode("tbd_cognet_stem"), device, feature_dim, module_dim, loaded_state_dict=stem_part)
    


    function_modules = {}  # holds our modules

    module_parts = []  


    # go through the vocab and add all the modules to our model
    for module_name in vocab['program_token_to_idx']:
        
        module_part = [(key,value) for key, value in load_obj.items() if module_name in key]
        module_parts.append((module_name, module_part))

        if module_name in ['<NULL>', '<START>', '<END>', '<UNK>', 'unique']:
            continue  # we don't need modules for the placeholders
        
        # figure out which module we want we use
        if module_name == 'scene':
            # scene is just a flag that indicates the start of a new line of reasoning
            # we set `module` to `None` because we still need the  'scene' flag in forward()
            module = None
        elif module_name == 'intersect':
            module =AndModule(ConceptNode("And_Module"))
        elif module_name == 'union':
            module = OrModule(ConceptNode("Or_Module"))
        elif 'equal' in module_name or module_name in {'less_than', 'greater_than'}:
            module = ComparisonModule(ConceptNode("Comparison_Module"),module_dim)
        elif 'query' in module_name or module_name in {'exist', 'count'}:
            module = QueryModule(ConceptNode("Query_Module"),module_dim)
        elif 'relate' in module_name:
            module = RelateModule(ConceptNode("Relate_Module"),module_dim)
        elif 'same' in module_name:
            module = SameModule(ConceptNode("Same_Module"),module_dim)
        else:
            module = AttentionModule(ConceptNode("Attention_Module"),module_dim)

        # add the module to our dictionary 
        # and register its parameters so it can learn
        function_modules[module_name] = module
        # add_module(module_name, module)
    
    torch.set_grad_enabled(False)

        

    def form_bindlink(atomspace, features, program, inheritance_set=None):

        print ("in form_bindlink we have")
        print ("atomspace = {}, program = {}".format(atomspace, program))
        
        # create a ConceptNode to hold our features
        bbox_instance = atomspace.add_node(types.ConceptNode, 'BoundingBox_instance')
       
        
        # now fill it with actual features
        set_value(bbox_instance, features)
                
        # create another ConceptNode, just for the BoundingBox concept
        box_concept = atomspace.add_node(types.ConceptNode, 'BoundingBox')
        
        # link BoundingBox and and instance of it. 
        # do we need this?
        atomspace.add_link(types.InheritanceLink, [bbox_instance, box_concept])



        current, rest = program[0], program[1:]
        
        if inheritance_set is None:
            inheritance_set = set()
        
        scene = atomspace.add_node(types.VariableNode, "$Scene")
        

        if current.startswith('query'):
            query_type = current.split('_')[-1]
            features_atom, attention_atom, left, inh = form_bindlink(atomspace, features, rest)
            print ("features_atom.execute() = {}".format(features_atom.execute()))
            print ("attention_atom = {}".format(attention_atom))
            sys.exit(0)

            inheritance_set |= inh
            var = atomspace.add_node(types.VariableNode, "$X")
            concept = atomspace.add_node(types.ConceptNode, query_type)
            inh_link = atomspace.add_link(types.InheritanceLink, [var, concept])
            assert(inh_link not in inheritance_set)
            inheritance_set.add(inh_link)
            # link = build_filter(atomspace, concept, var, exec_out_sub=sub_prog)
            

            module_type = 'filter_' + query_type + '[' + var + ']'
            module = function_modules[module_type]

            link = module.execute(features_atom.execute(), attention_atom)


            varlist = []
            for inh in inheritance_set:
                for atom in inh.get_out():
                    if atom.type == types.VariableNode:
                        varlist.append(atom)

            print ("varlist = {}".format(varlist))
            print ("link = {}".format(link))
                    
            
            variable_list = atomspace.add_link(types.VariableList, varlist)
            conj = atomspace.add_link(types.AndLink, [*inheritance_set])
            print ("conj = {}".format(conj))
            list_link = atomspace.add_link(types.ListLink, varlist + [link])
            print ("list_link = {}".format(list_link))
                       
           
            bind_link_0 = BindLink(variable_list, conj, list_link)
            print ("bind_link_0 = {}".format(bind_link_0))
            # sys.exit(0)


            module_type = 'filter_' + query_type# + '[' + var + ']'
            module = function_modules[module_type]

            return link, left, inheritance_set
        

        elif current.startswith('scene'):
            concept = box_concept
            inh_link = atomspace.add_link(types.InheritanceLink, [scene, concept])
            inheritance_set.add(inh_link)
            
                
            atomspace = scene.atomspace
            
            # let's turn data_atom into an InputModule
            # it holds features and attention
            # as a list of 2 tensors
            # TODO: find a better way to deal with this
            # use two distinct atoms
            features_atom = InputModule(ConceptNode("Data-{}".format(str(uuid.uuid4()))), features)
            attention_atom = InputModule(ConceptNode("Attention-{}".format(str(uuid.uuid4()))), ones_var)

            return features_atom, attention_atom, rest, inheritance_set
        

        elif current.startswith('filter'):
            print ("in filter branch, we have current {}".format(current))
                
                
            filter_type, filter_arg = filter_reg.match(current).groups()
            features_atom, attention_atom, left, inh = form_bindlink(atomspace, features, rest)
            # print ("in filter branch, sub_prog = {}".format(sub_prog))
            
            filter_type_atom = atomspace.add_node(types.ConceptNode, filter_type)
            filter_arg_atom = atomspace.add_node(types.ConceptNode, filter_arg)
            
           
            inheritance_set |= inh
            
            # data_atom = sub_prog
            print ("we have filter_type {} and filter_arg {}".format(filter_type_atom.name, filter_arg_atom.name))
            inh_filter = InheritanceLink(filter_arg_atom, filter_type_atom)
            print ("InheritanceLink was created:{}".format(inh_filter))
            # TODO: now that we have an inheritance link, 
            # how do we use it?
            
            
            module_type = 'filter_' + filter_type_atom.name + '[' + filter_arg_atom.name + ']'
            module = function_modules[module_type]
         
            print ("features_atom.execute() = {}".format(features_atom.execute()))
            if isinstance(attention_atom, CogModule):
                print ("attention_atom.execute() = {}".format(attention_atom.execute()))
            else:
                print ("attention_atom = {}".format(attention_atom))
            # out = module.execute() #data_atom.execute()[0], data_atom.execute()[1]
            if isinstance(attention_atom, CogModule):
                out = module.execute(features_atom.execute(), attention_atom.execute())
            else:
                out = module.execute(features_atom.execute(), attention_atom)
            print ("out = {}".format(out))
            # sys.exit(0)

            # this is probably a bad way to set values
            # but it works for now
            # TODO: find a better way
            # data_atom()[1] = out
            # print ("after, data_atom = {}".format(data_atom()))
            attention_atom = out
           

            return features_atom, attention_atom, left, inheritance_set    

        elif current.startswith('relate'):
            relate_arg = relate_reg.match(current).groups()[0]
            sub_prog, left, inh = form_bindlink(atomspace, rest)
            inheritance_set |= inh
            return build_relate(atomspace, relate_argument=relate_arg,
                                exec_out_sub=sub_prog), left, inheritance_set
        
        elif current.startswith('same'):
            same_arg = current.split('_')[-1]
            sub_prog, left, inh = form_bindlink(atomspace, rest)
            inheritance_set |= inh
            return build_same(atomspace, same_argument=same_arg,
                              exec_out_sub=sub_prog), left, inheritance_set
        
        elif current.startswith('intersect'):
            sub_prog0, left, inh = form_bindlink(atomspace, features, rest)
            inheritance_set |= inh
            sub_prog1, right, inh = form_bindlink(atomspace, features, left)
            inheritance_set |= inh
            return build_intersect(atomspace, arg0=sub_prog0, arg1=sub_prog1), right, inheritance_set
        
        elif current == '<START>':
            return form_bindlink(atomspace, features, rest)
        
        elif current == 'unique':
            return form_bindlink(atomspace, features, rest)
        
        else:
            raise NotImplementedError(current)

           

    BATCH_SIZE = 64
    h5_path = Path('/media/enoch/0645F864324D53D4/neural_stuff/CLEVR_v1/data/')
    h5_files = [x for x in h5_path.iterdir() if x.is_file()]
    
    val_loader_kwargs = {
        'question_h5':h5_files[1], 
        'feature_h5': h5_files[0], 
        'batch_size': BATCH_SIZE,
        'num_workers': 1,
        'shuffle': False
    }


    loader = ClevrDataLoaderH5(**val_loader_kwargs)
    
    for i, batch in enumerate(tqdm(loader)):
        print ("working with batch #{}".format(i))

        _, _, feats, expected_answers, programs = batch
        feats = feats.to(device)

        feats_module = InputModule(ConceptNode("batch_features"), feats)

        programs = programs.to(device)

        features = feats_module

        batch_size = features().size(0)

               
        for n in range(batch_size):
            
            output = stem(features())[n:n + 1]

            program_list = [vocab['program_idx_to_token'][i] \
                            for i in reversed(programs.data[n].cpu().numpy()) \
                            if vocab['program_idx_to_token'][i] != '<NULL>']    
        
            rev_prog = tuple(reversed(program_list))

            eval_link, left, inheritance_set = form_bindlink(atomspace, output, rev_prog)
            sys.exit(0)
              

if __name__ == '__main__':
    main()
