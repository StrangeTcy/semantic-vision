from pathlib import Path
import torch

from tbd.tbd.module_net_cog import load_tbd_net
from tbd_cog import tbd_vqa
from util import initialize_atomspace_by_facts
from tbd.utils.clevr import load_vocab, ClevrDataLoaderH5
from tbd_cog import tbd_helpers

# imports to be used as callbacks in atomspace
from tbd_cog.tbd_helpers import init_scene, filter_callback, intersect, relate, same

import os
import sys

from tqdm import tqdm #nice progress bars


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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    atomspace = initialize_atomspace_by_facts("tbd_cog/tbdas.scm")
    tbd = tbd_vqa.TransparentByDesignVQA(None, None, atomspace, None)
    tbd_net_checkpoint = Path('/media/enoch/0645F864324D53D4/neural_stuff/tbd/')
    files = [x for x in tbd_net_checkpoint.iterdir() if x.is_file()]
    # print (files) 
    # print ("tbd_net_checkpoint = {}".format(tbd_net_checkpoint))
    # if tbd_net_checkpoint.is_file():
    #     print ("Ok, file seems to exist")
    # else:
    #     print ("Something went wrong")
    #     sys.exit(0)
    tbd_net_checkpoint = files[0]
    # tbd_net_checkpoint = '/media/enoch/0645F864324D53D4/neural_stuff/tbd/сlevr-reg.pt' 
    #'/mnt/fileserver/shared/models/tbd-nets-models/clevr-reg.pt'
    # tbd_net_checkpoint = Path('/home/enoch/Desktop/сlevr-reg.pt')
    vocab = load_vocab(Path('/media/enoch/0645F864324D53D4/neural_stuff/tbd/data/vocab.json'))  #'/mnt/fileserver/shared/models/tbd-nets-models/data/vocab.json'
    tdb_net = load_tbd_net(device, tbd_net_checkpoint, vocab, feature_dim=(1024, 14, 14) )  
    # print ("tdb_net={}".format(tdb_net))
    # sys.exit(0)
    BATCH_SIZE = 64
    h5_path = Path('/media/enoch/0645F864324D53D4/neural_stuff/CLEVR_v1/data/')
    h5_files = [x for x in h5_path.iterdir() if x.is_file()]
    print (h5_files)
    val_loader_kwargs = {
        'question_h5':h5_files[1], #'/mnt/fileserver/shared/datasets/CLEVR_v1/data/val_questions_query_ending.h5'
        'feature_h5': h5_files[0], #'/mnt/fileserver/shared/datasets/CLEVR_v1/data/val_features.h5'
        'batch_size': BATCH_SIZE,
        'num_workers': 1,
        'shuffle': False
    }

    tbd_helpers.tbd = tdb_net
    loader = ClevrDataLoaderH5(**val_loader_kwargs)
    total_acc = 0
    for i, batch in enumerate(tqdm(loader)):
        _, _, feats, expected_answers, programs = batch
        feats = feats.to(device)
        programs = programs.to(device)

        # print ("About to call tbd.answer_by_programs with tdb_net = {}".format(tdb_net))
        # print (", feats= {}".format(feats))
        # print ("programs={}".format(programs))
        results = tbd.answer_by_programs(tdb_net, feats, programs)
        correct = 0
        clevr_numeric_actual_answers = [clevr_answers_map_str_int[x] for x in results]
        for (actual, expected) in zip(clevr_numeric_actual_answers, expected_answers):
            correct += 1 if actual == expected else 0
        acc = float(correct) / len(programs)
        total_acc = total_acc * (i / (i + 1)) + acc/(i + 1)
        print("Accuracy average: {:.4f}%".format(total_acc*100.0))


if __name__ == '__main__':
    main()
