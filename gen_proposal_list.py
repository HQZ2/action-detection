import argparse
import pickle
import os
from ops.io import process_proposal_list, parse_directory
from ops.utils import get_configs


parser = argparse.ArgumentParser(
    description="Generate proposal list to be used for training")
parser.add_argument('dataset', type=str, choices=['activitynet1.2', 'thumos14'])
parser.add_argument('rgb_path', type=str)
parser.add_argument('flow_path', type=str)

args = parser.parse_args()

configs = get_configs(args.dataset)

norm_list_tmpl = 'data/{}_normalized_proposal_list.txt'
out_list_tmpl = 'data/{}_proposal_list.txt'


if args.dataset == 'activitynet1.2':
    key_func = lambda x: x[-15:-4]
elif args.dataset == 'thumos14':
    key_func = lambda x: x.split('/')[-1]
else:
    raise ValueError("unknown dataset {}".format(args.dataset))


# parse the folders holding the extracted frames
#frame_dict = parse_directory(args.rgb_path, args.flow_path, key_func=key_func)
pkl = open('data/frame_dict.pkl','rb')
frame_dict = pickle.load(pkl)

process_proposal_list(norm_list_tmpl.format(configs['train_list']),
                      out_list_tmpl.format(configs['train_list']), frame_dict)
process_proposal_list(norm_list_tmpl.format(configs['test_list']),
                      out_list_tmpl.format(configs['test_list']), frame_dict)

print("proposal lists for dataset {} are ready for training.".format(args.dataset))
