from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F  


from tbd.utils.clevr import load_vocab, ClevrDataLoaderH5

import os
import sys

import numpy as np

from tqdm import tqdm #nice progress bars

from opencog.type_constructors import *
from opencog.utilities import initialize_opencog, finalize_opencog

from opencog.scheme_wrapper import scheme_eval_as, scheme_eval

from tbd.tbd import cog  # has from opencog.atomspace import types
                         #     from opencog.type_constructors import *   

from pattern_matcher_vqa import PatternMatcherVqaPipeline

from opencog.bindlink import execute_atom



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


#============================================================


def pushAtomspace(parentAtomspace):
    """Create child atomspace"""
    # TODO: cannot push/pop atomspace via Python API,
    # working around it using Scheme API
    scheme_eval(parentAtomspace, '(cog-push-atomspace)')
    childAtomspace = scheme_eval_as('(cog-atomspace)')
    set_type_ctor_atomspace(childAtomspace)
    return childAtomspace


def popAtomspace(childAtomspace):
    """Destroy child atomspace"""
    scheme_eval(childAtomspace, '(cog-pop-atomspace)')
    parentAtomspace = scheme_eval_as('(cog-atomspace)')
    set_type_ctor_atomspace(parentAtomspace)
    return parentAtomspace



#============================================================


def initialize_atomspace_by_facts(atomspaceFileName=None, ure_config=None, directories=[]):
    atomspace = scheme_eval_as('(cog-atomspace)')
    scheme_eval(atomspace, '(use-modules (opencog))')
    scheme_eval(atomspace, '(use-modules (opencog exec))')
    scheme_eval(atomspace, '(use-modules (opencog query))')
    scheme_eval(atomspace, '(use-modules (opencog logger))')
    scheme_eval(atomspace, '(add-to-load-path ".")')
    for item in directories:
        scheme_eval(atomspace, '(add-to-load-path "{0}")'.format(item))
    if atomspaceFileName is not None:
        scheme_eval(atomspace, '(load-from-path "' + atomspaceFileName + '")')
    if ure_config is not None:
        scheme_eval(atomspace, '(load-from-path "' + ure_config + '")')
    print ("facts were loaded into atomspace {}".format(atomspace))
    return atomspace

#============================================================

"""
Converters for tbd programs to atomese and callbacks
"""

import re
import uuid

relate_reg = re.compile('[a-z]+\[([a-z]+)\]')
filter_reg = re.compile('[a-z]+_([a-z]+)\[([a-z]+)\]')
filter_reg.match('filter_shape[sphere]').groups()
relate_reg.match('relate[front]').groups()



def build_eval_link(atomspace, classify_type, variable, eval_link_sub):
    predicate =  atomspace.add_node(types.GroundedSchemaNode, "py:classify")
    variable = atomspace.add_node(types.VariableNode, "$X")
    lst = atomspace.add_link(types.ListLink, [variable, eval_link_sub])
    return atomspace.add_link(types.EvaluationLink, [predicate, lst])

#===========================================================


# CALLBACKS


def extract_tensor(atom, key_data, key_shape):
    """
    Convert FloatValue attached to atom to pytorch array
    :param atom:
    :param key_data:
    :param key_shape:
    :return: torch.Tensor
    """
    
    shape = atom.get_value(key_shape)
        
    value = atom.get_value(key_data)
    
    value_tlst = value.to_list()
        
    ar_value = np.array(value_tlst)
      
    ar_value = ar_value.reshape([int(x) for x in shape.to_list()])
   
    tensor_result = torch.from_numpy(ar_value).double()
   
    return tensor_result


def generate_keys(atomspace):
    """
    Return predicate nodes 
    to be used as keys for attaching values to atoms
    :param atomspace: AtomSpace
    :return: Tuple[Atom]
    """
    
    key_scene = atomspace.add_node(types.PredicateNode, 'key_scene')
    key_shape_scene = atomspace.add_node(types.PredicateNode, 'key_shape_scene')
    key_attention = atomspace.add_node(types.PredicateNode, 'key_data')
    key_shape_attention = atomspace.add_node(types.PredicateNode, 'key_shape_attention')
       
    return key_attention, key_scene, key_shape_attention, key_shape_scene


def init_scene(scene):
    """
    Accept scene atom 
    and generate new atom which holds 
    dummy attention map
    and features from scene
    :param scene: Atom
    :return: Atom
        An atom with features, attention map, 
        and size for features and attention map
    """
    
    atomspace = scene.atomspace
   
    data_atom = atomspace.add_node(types.ConceptNode, 'Data-' + str(uuid.uuid4()))
    key_attention, key_scene, key_shape, key_shape_scene = generate_keys(atomspace)

    
    data_atom.set_value(key_scene, scene.get_value(key_scene))
    data_atom.set_value(key_shape_scene, scene.get_value(key_shape_scene))
    data_atom.set_value(key_attention, FloatValue(list(tbd.ones_var.flatten())))
    data_atom.set_value(key_shape, FloatValue(list(tbd.ones_var.shape)))
    return data_atom


def filter_callback(filter_type, filter_type_instance, data_atom):
    
    """
    Function which applies 
    the filtering neural network module

    :param filter_type: Atom
        An atom with name of filter type e.g. color or size
    :param filter_type_instance: Atom
        An atom with name of particular filter instance e.g. red or small
    :param data_atom:
        An atom with attention map and features attached
    :return:
    """
       
    module_type = 'filter_' + filter_type.name + '[' + filter_type_instance.name + ']'
    run_attention(data_atom, module_type)
    return data_atom


def set_attention_map(data_atom, key_attention, key_shape_attention, attention):
    
    """
    Attach attention map to atom

    :param data_atom: Atom
    :param key_attention: Atom
        Atom to be used as key for the attention map
    :param key_shape_attention:
        Atom to be used as key for the attention map shape
    :param attention: numpy.array
    :return:
    """
    
    data_atom.set_value(key_attention, FloatValue(list(attention.flatten())))
    data_atom.set_value(key_shape_attention, FloatValue(list(attention.shape)))


def intersect(arg0, arg1):
    """
    Intersection of attention maps

    :param arg0: Atom
        An atom with attention map and features attached
    :param arg1: Atom
        An atom with attention map and features attached
    :return: Atom
        arg0 with new attention map attached
    """
    
    atomspace = arg0.atomspace
    key_attention, key_scene, key_shape_attention, key_shape_scene = generate_keys(atomspace)
    feat_attention1 = extract_tensor(arg0, key_attention, key_shape_attention)
    feat_attention2 = extract_tensor(arg1, key_attention, key_shape_attention)
    module = tbd.function_modules['intersect']
    out = module(feat_attention1, feat_attention2)
    set_attention_map(arg0, key_attention, key_shape_attention, out)
    return arg0


def classify(classifier_type, instance, data_atom):
    """
    Same as filter_callback : 
    should be replaced with classifier returning tv
    """
    return filter_callback(classifier_type, instance, data_atom)


def relate(relation, data_atom):
    """
    Function which applies filtering neural network module

    :param relation: Atom
        An atom with name of type of relation e.g. front or left etc.
    :param data_atom: Atom
        An atom with attention map and features attached
    """

    module_type = 'relate[' + relation.name + ']'
    run_attention(data_atom, module_type)
    return data_atom


def same(relation, data_atom):
    """
    Function which applies same neural network module

    :param relation: Atom
        An atom with name of type of 'same' relation e.g. same color or size etc.
    :param data_atom: Atom
        Atom with attention map and features
    :return:
    """
    module_type = 'same_' + relation.name
    run_attention(data_atom, module_type)
    return data_atom


def run_attention(data_atom, module_type):
    """
    Run neural network module which accepts attention map and features
    and produces attention map

    :param data_atom: Atom
        An atom with attached attention map and features
    :param module_type: str
        Module type name: e.g. filter_color[red] or same_size
    :return:
    """

    
    module = tbd.function_modules[module_type]
    atomspace = data_atom.atomspace
   
    key_attention, key_scene, key_shape_attention, key_shape_scene = generate_keys(atomspace)
    
    feat_input = extract_tensor(data_atom, key_scene, key_shape_scene)
   
    feat_attention = extract_tensor(data_atom, key_attention, key_shape_attention)
    
    out = module(feat_input.float(), feat_attention.float())
    set_attention_map(data_atom, key_attention, key_shape_attention, out)

#============================================================


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Stem(cog.cogModule):
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
         

    def forward(self,x):
        return self.stem(x) 


class Classifier(cog.cogModule):
    """
    The classifier takes the output of the last module 
    (which will be a Query or Equal module)
    and produces a distribution over answers
    """
    def __init__(self, atom, device, module_dim, cls_proj_dim, module_rows, module_cols, fc_dim, loaded_state_dict):
        super().__init__(atom)
        self.classifier = nn.Sequential(nn.Conv2d(module_dim, cls_proj_dim, kernel_size=1),
                                        nn.ReLU(inplace=True),
                                        nn.MaxPool2d(kernel_size=2, stride=2),
                                        Flatten(),
                                        nn.Linear(cls_proj_dim * module_rows * module_cols // 4,
                                                  fc_dim),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(fc_dim, 28)  # note no softmax here
                                       )
        self.classifier = self.classifier.to(device)

        self_class_part = [key for key, value in self.classifier.state_dict().items() ] 
        if not loaded_state_dict == None:
            for _, (key, value) in enumerate(self.classifier.state_dict().items()):
                value = loaded_state_dict[_][1]


        
    def forward(self,x):
        return self.classifier(x)         



class AndModule(cog.cogModule):
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
    
    def __init__(self, atom):
        super().__init__(atom)
    

    def forward(self, attn1, attn2):
        out = torch.min(attn1, attn2)
        return out


class OrModule(cog.cogModule):
    """ A neural module that (basically) performs a logical or.

    Extended Summary
    ----------------
    An :class:`OrModule` is a neural module that takes two input attention masks and (basically)
    performs a set union. This would be used in a question like "How many cubes are left of the
    brown sphere or right of the cylinder?" After localizing the regions left of the brown sphere
    and right of the cylinder, an :class:`OrModule` would be used to find the union of the two. Its
    output would then go into an :class:`AttentionModule` that finds cubes.
    """

    def __init__(self, atom):
        super().__init__(atom)
      

    def forward(self, attn1, attn2):
        out = torch.max(attn1, attn2)
        return out



class AttentionModule(cog.cogModule):
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


class QueryModule(cog.cogModule):
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


class RelateModule(cog.cogModule):
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


class SameModule(cog.cogModule):
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


class ComparisonModule(cog.cogModule):
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

#========================================================


class TBD(cog.cogModule):
    
    """ 
    The real deal. 
    A full Transparency by Design network (TbD-net).
    This is wrapped in a cogModule 
    and so should be called and executed as one

    TODO: re-write training in opencog/atomese 

    Extended Summary
    ----------------
    A :class:`TbDNet` holds neural :mod:`modules`, 
    a stem network, and a classifier network. 
    It hooks these all together 
    to answer a question
    given some scene 
    and a program describing 
    how to arrange the neural modules.
    """
    
    def __init__(self,
                 atom,
                 atomspace,
                 device,
                 vocab,
                 checkpoint,
                 feature_dim=(512, 28, 28),
                 module_dim=128,
                 cls_proj_dim=512,
                 fc_dim=1024, 
                 load_checkpoint=True):
        
        """ 
        Initializes a TbDNet object.

        Parameters
        ----------
        atom: a ConceptNode with a name of the whole net

        vocab : Dict[str, Dict[Any, Any]]
            The vocabulary holds dictionaries that provide handles to various objects. Valid keys 
            into vocab are
            - 'answer_idx_to_token' whose keys are ints and values strings
            - 'answer_token_to_idx' whose keys are strings and values ints
            - 'program_idx_to_token' whose keys are ints and values strings
            - 'program_token_to_idx' whose keys are strings and values ints
            These value dictionaries provide retrieval of an answer word or program token from an
            index, or an index from a word or program token.

        feature_dim : the tuple (K, R, C), optional
            The shape of input feature tensors, excluding the batch size.

        module_dim : int, optional
            The depth of each neural module's convolutional blocks.

        cls_proj_dim : int, optional
            The depth to project the final feature map to before classification.
        """

        super().__init__(atom)
        
        
        self.atomspace = atomspace
        initialize_opencog(self.atomspace)

        self.device = device
        
        module_rows, module_cols = feature_dim[1], feature_dim[2]

        
        load_obj = torch.load(checkpoint)
        load_obj_keys = [key for key,value in load_obj.items()]
        

        if load_checkpoint:
            str_checkpoint = str(checkpoint)
            load_obj = torch.load(checkpoint) 

                       
            stem_part = [(key,value) for key, value in load_obj.items() if "stem" in key]
            self.stem = Stem(ConceptNode("tbd_cognet_stem"), self.device, feature_dim, module_dim, loaded_state_dict=stem_part)
            class_part = [(key,value) for key, value in load_obj.items() if "classifier" in key]
            self.classifier = Classifier(ConceptNode("tbd_cognet_stem"), device, module_dim, cls_proj_dim, module_rows, module_cols, fc_dim, loaded_state_dict=class_part)
        else:

            self.stem = Stem(ConceptNode("tbd_cognet_stem"), self.device, feature_dim, module_dim, loaded_state_dict=None)
            self.classifier = Classifier(ConceptNode("tbd_cognet_stem"), device, module_dim, cls_proj_dim, module_rows, module_cols, fc_dim, loaded_state_dict=None)

            
        
        self.vocab = vocab
        self.function_modules = {}  # holds our modules

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
            self.function_modules[module_name] = module
            self.add_module(module_name, module)

               
        
        # this part loads module parameters from a checkpoint
        if load_checkpoint:
            for module_name in vocab['program_token_to_idx']:
                print ("working with module_name {}".format(module_name))
                if module_name in ['<NULL>', '<START>', '<END>', '<UNK>', 'unique']:
                    print ("skipping...")
                    continue
                
                thing1 = [load_obj[name] for name in load_obj_keys if module_name in name] 
                thing2 = [self.state_dict()[name] for name in load_obj_keys if  module_name in name]
                print ("replacing...")
                thing2 = thing1
                thing2 = [self.state_dict()[name] for name in load_obj_keys if  module_name in name]
                      
        
        # this is used as input to the first AttentionModule in each program
        ones = torch.ones(1, 1, module_rows, module_cols)
        self.ones_var = ones.to(self.device)
        self._attention_sum = 0
        

        self.features = None # initialise with none, then add values
        self = self.to(device)
    

    @property
    def attention_sum(self):
        
        '''
        Returns
        -------
        attention_sum : int
            The sum of attention masks produced during the previous forward pass, or zero if a
            forward pass has not yet happened.

        Extended Summary
        ----------------
        This property holds the sum of attention masks produced during a forward pass of the model.
        It will hold the sum of all the AttentionModule, RelateModule, and SameModule outputs. This
        can be used to regularize the output attention masks, hinting to the model that spurious
        activations that do not correspond to objects of interest (e.g. activations in the 
        background) should be minimized. For example, a small factor multiplied by this could be
        added to your loss function to add this type of regularization as in:

            loss = xent_loss(outs, answers)
            loss += executor.attention_sum * 2.5e-07
            loss.backward()

        where `xent_loss` is our loss function, `outs` is the output of the model, `answers` is the
        PyTorch `Tensor` containing the answers, and `executor` is this model. The above block
        will penalize the model's attention outputs multiplied by a factor of 2.5e-07 to push the
        model to produce sensible, minimal activations.
        '''
        
        return self._attention_sum

    
    def answer_by_programs(self, features, programs):
        """
        Compute answers from image features and programs
        
        :param features: torch.Tensor
            Images features
        :param programs: torch.Tensor
            Programs in numeric form
        :return: List[str]
            answers as strings
        """

        batch_size = features.size(0)
        feat_input_volume = self.stem(features)
       
        results2 = []
        
        for n in range(batch_size): #[:4]
            feat_input = feat_input_volume[n:n + 1]
            output = feat_input
            program = []
            for i in reversed(programs.data[n].cpu().numpy()):
                module_type = self.vocab['program_idx_to_token'][i]
                if module_type == '<NULL>':
                    continue
                program.append(module_type)
            

            result2 = self.run_program(output, program)
            results2.append(result2)

        return results2


    def run_program(self, features, program):
        self.atomspace = pushAtomspace(self.atomspace)
        
        _, key_scene, _, key_shape_scene = generate_keys(self.atomspace)
        

        self.features = features

       
        rev_prog = tuple(reversed(program))
        
                
        print ("we'll now pass it to return_prog2")
        eval_link2, left2, inheritance_set2, answer_stuff = self.return_prog2(self.atomspace, rev_prog)
        
        
        #======================================
        
        result2 = answer_stuff
        answer_set2 = result2
        items2 = []


        print ("argmax-ing...")
        key_attention, key_scene, key_shape_attention, key_shape_scene = generate_keys(self.atomspace)
        for list_link in answer_set2.get_out():
            atoms = list_link.get_out()
            value = -1
            concept = None
            for atom in atoms:
                if atom.name.startswith("Data-"):
                    value_ = extract_tensor(atom, key_attention, key_shape_attention)
                    if str(value_.device).startswith("cuda"):
                        value_ = value_.to("cpu")
                    value = value_.numpy().sum()
                elif atom.name.startswith("BoundingBox"):
                    continue
                else:
                    concept = atom.name
            assert concept
            assert value != -1
            items2.append((value, concept))
        if not items2:
            return None
        items2.sort(reverse=True)
        answer2 = items2[0][1]

        print ("answer2 = {}".format(answer2))


       
        self.atomspace = popAtomspace(self.atomspace)
        return answer2    


    
    def return_prog2(self, atomspace, commands, inheritance_set2=None):
        
        
        _, key_scene, _, key_shape_scene = generate_keys(self.atomspace)
        
        features = self.features

        data2 = FloatValue(list(features.cpu().numpy().flatten()))
        bbox_instance2 = self.atomspace.add_node(types.ConceptNode, 'BoundingBox1')
        bbox_instance2.set_value(key_scene, data2)
        bbox_instance2.set_value(key_shape_scene, FloatValue(list(features.cpu().numpy().shape)))
        box_concept2 = self.atomspace.add_node(types.ConceptNode, 'BoundingBox')
        self.atomspace.add_link(types.InheritanceLink, [bbox_instance2, box_concept2])
    

        current, rest = commands[0], commands[1:]
        if inheritance_set2 is None:
            inheritance_set2 = set()
        scene = atomspace.add_node(types.VariableNode, "$Scene")
        
        schema = None
        lst = None
        left2 = None
        final_stuff  = None


        def final_exec(inheritance_set_2, link):
           
            varlist2 = []
            for inh in inheritance_set_2:
                for atom in inh.get_out():
                    if atom.type == types.VariableNode:
                        varlist2.append(atom)
            
            variable_list2 = self.atomspace.add_link(types.VariableList, varlist2)
            list_link2 = self.atomspace.add_link(types.ListLink, varlist2 + [link])
            conj2 = self.atomspace.add_link(types.AndLink, [*inheritance_set_2]) #, link
            
            bind_link_0 = self.atomspace.add_link(types.BindLink, [variable_list2, conj2, list_link2])
            stuff_5_0 = execute_atom(self.atomspace, bind_link_0)
            return stuff_5_0          

        

        if current.startswith('query'):
            query_type = current.split('_')[-1]
            sub_prog, left2, inh, final_stuff = self.return_prog2(atomspace, rest)
            inheritance_set2 |= inh
            var = atomspace.add_node(types.VariableNode, "$X")
            concept = atomspace.add_node(types.ConceptNode, query_type)
            inh_link = atomspace.add_link(types.InheritanceLink, [var, concept])
            assert(inh_link not in inheritance_set2)
            inheritance_set2.add(inh_link)
            
            schema = atomspace.add_node(types.GroundedSchemaNode, "py:filter_callback")

            lst = atomspace.add_link(types.ListLink, [concept,
                                                  var,
                                                  sub_prog])
            link = atomspace.add_link(types.ExecutionOutputLink, [schema, lst])


            if left2 == ('<END>',):
                                
                final_stuff = final_exec(inheritance_set2, link)
                

            exec_result_query = None  
            return exec_result_query, left2, inheritance_set2, final_stuff

            
        
        elif current.startswith('scene'):
            concept = box_concept2
            inh_link = atomspace.add_link(types.InheritanceLink, [scene, concept])
            inheritance_set2.add(inh_link)
            
            scene = bbox_instance2
           
            atomspace = scene.atomspace
            
            data_atom = atomspace.add_node(types.ConceptNode, 'Data-' + str(uuid.uuid4()))
            key_attention, key_scene, key_shape, key_shape_scene = generate_keys(atomspace)

            
            data_atom.set_value(key_scene, scene.get_value(key_scene))
            data_atom.set_value(key_shape_scene, scene.get_value(key_shape_scene))
            data_atom.set_value(key_attention, FloatValue(list(self.ones_var.flatten())))
            data_atom.set_value(key_shape, FloatValue(list(self.ones_var.shape)))
            return data_atom, rest, inheritance_set2, final_stuff
        
        elif current.startswith('filter'):
            
            filter_type, filter_arg = filter_reg.match(current).groups()
            sub_prog, left2, inh, final_stuff = self.return_prog2(self.atomspace, rest)
            filter_type_atom = atomspace.add_node(types.ConceptNode, filter_type)
            filter_arg_atom = atomspace.add_node(types.ConceptNode, filter_arg)
            
           
            inheritance_set2 |= inh
            

            print ("using filter modules directly:")

            data_atom = sub_prog
            module_type = 'filter_' + filter_type_atom.name + '[' + filter_arg_atom.name + ']'
            module = self.function_modules[module_type]
           
            atomspace = data_atom.atomspace
            key_attention, key_scene, key_shape_attention, key_shape_scene = generate_keys(atomspace)
            feat_input = extract_tensor(data_atom, key_scene, key_shape_scene)
            feat_attention = extract_tensor(data_atom, key_attention, key_shape_attention)
                        
            out = module(feat_input.float(), feat_attention.float())
            set_attention_map(data_atom, key_attention, key_shape_attention, out)

            return data_atom, left2, inheritance_set2, final_stuff    
            



        elif current.startswith('relate'):
            relate_arg = relate_reg.match(current).groups()[0]
            sub_prog, left2, inh, final_stuff = self.return_prog2(atomspace, rest)
            inheritance_set2 |= inh
            
            
            data_atom = sub_prog
            module_type = 'relate[' + atomspace.add_node(types.ConceptNode, relate_arg).name+']'
            run_attention(data_atom, module_type) #might replace that as well
            return data_atom, left2, inheritance_set2, final_stuff


        elif current.startswith('same'):
            same_arg = current.split('_')[-1]
            sub_prog, left2, inh, final_stuff = self.return_prog2(atomspace, rest)
            inheritance_set2 |= inh
            
             
            data_atom = sub_prog
            module_type = 'same_' + atomspace.add_node(types.ConceptNode, same_arg).name
            run_attention(data_atom, module_type) #might need to replace that
            return data_atom, left2, inheritance_set2, final_stuff



        elif current.startswith('intersect'):
            sub_prog0, left, inh, final_stuff = self.return_prog2(atomspace, rest)
            inheritance_set2 |= inh
            sub_prog1, right, inh, final_stuff = self.return_prog2(atomspace, left)
            inheritance_set2 |= inh
            
            
            print ("using the intersect module directly:")
            
            atomspace = sub_prog0.atomspace
            key_attention, key_scene, key_shape_attention, key_shape_scene = generate_keys(atomspace)
            feat_attention1 = extract_tensor(sub_prog0, key_attention, key_shape_attention)
            feat_attention2 = extract_tensor(sub_prog1, key_attention, key_shape_attention)
            module = self.function_modules['intersect']
            out = module(feat_attention1, feat_attention2)
            set_attention_map(sub_prog0, key_attention, key_shape_attention, out)
            return sub_prog0, right, inheritance_set2, final_stuff


        elif current == '<START>':
            return self.return_prog2(atomspace, rest)
        
        elif current == 'unique':
            return self.return_prog2(atomspace, rest)

        else:
            raise NotImplementedError(current)      



def main():
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    atomspace = initialize_atomspace_by_facts("tbd_cog/tbdas.scm")
    
    tbd_net_checkpoint = Path('/media/enoch/0645F864324D53D4/neural_stuff/tbd/')
    files = [x for x in tbd_net_checkpoint.iterdir() if x.is_file()]
    
    tbd_net_checkpoint = files[0]
    vocab = load_vocab(Path('/media/enoch/0645F864324D53D4/neural_stuff/tbd/data/vocab.json'))  #'/mnt/fileserver/shared/models/tbd-nets-models/data/vocab.json'
    

    # we have to initialize atomspace here, or Node creation won't work
    
    initialize_opencog(atomspace)

    global tbd 

    tbd = TBD(ConceptNode("loaded_tbd_cognet"), atomspace, device,  vocab, tbd_net_checkpoint, feature_dim=(1024, 14, 14), load_checkpoint=True)
    
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
    total_acc2 = 0
    for i, batch in enumerate(tqdm(loader)):
        print ("working with batch #{}".format(i))

        _, _, feats, expected_answers, programs = batch
        feats = feats.to(device)
        programs = programs.to(device)
        
        results2 = tbd.answer_by_programs(feats, programs)
        
        correct2 = 0
        clevr_numeric_actual_answers2 = [clevr_answers_map_str_int[x] for x in results2]
        for (actual, expected) in zip(clevr_numeric_actual_answers2, expected_answers):
            correct2 += 1 if actual == expected else 0
        acc2 = float(correct2) / len(programs)
        total_acc2 = total_acc2 * (i / (i + 1)) + acc2/(i + 1)
        print("Accuracy average2: {:.4f}%".format(total_acc2*100.0))
        # if i >= 4:
        #     print ("that's enough for now")
        #     sys.exit(0)        

if __name__ == '__main__':
    main()
