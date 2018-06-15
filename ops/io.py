import numpy as np
import glob
import os
import fnmatch


def load_proposal_file(filename):
    lines = list(open(filename))
    from itertools import groupby
    groups = groupby(lines, lambda x: x.startswith('#'))

    info_list = [[x.strip() for x in list(g)] for k, g in groups if not k]
    
    def parse_group(info):
        offset = 0
        vid = info[offset]
        offset += 1

        n_frame = int(float(info[1]) * float(info[2]))
        n_gt = int(info[3])
        offset = 4

        gt_boxes = [x.split() for x in info[offset:offset+n_gt]]
        offset += n_gt
        if n_gt == 0:
            offset += 1
        n_pr = int(info[offset])
        offset += 1
        pr_boxes = [x.split() for x in info[offset:offset+n_pr]]

        return vid, n_frame, gt_boxes, pr_boxes

    return [parse_group(l) for l in info_list]


def process_proposal_list(norm_proposal_list, out_list_name, frame_dict):
    norm_proposals = load_proposal_file(norm_proposal_list)

    processed_proposal_list = []
    for idx, prop in enumerate(norm_proposals):
        vid = prop[0]
        try:
            frame_info = frame_dict[vid]
        except:
            print('{} is invalid!'.format(vid))
            continue
        frame_cnt = frame_info[1]
        frame_path = frame_info[0]

        gt = [[int(x[0]), int(float(x[1]) * frame_cnt), int(float(x[2]) * frame_cnt)] for x in prop[2]]

        prop = [[int(x[0]), float(x[1]), float(x[2]), int(float(x[3]) * frame_cnt), int(float(x[4]) * frame_cnt)] for x
                in prop[3]]

        out_tmpl = "# {idx}\n{path}\n{fc}\n1\n{num_gt}\n{gt}{num_prop}\n{prop}"

        gt_dump = '\n'.join(['{} {:d} {:d}'.format(*x) for x in gt]) + ('\n' if len(gt) else '')
        prop_dump = '\n'.join(['{} {:.04f} {:.04f} {:d} {:d}'.format(*x) for x in prop]) + (
            '\n' if len(prop) else '')

        processed_proposal_list.append(out_tmpl.format(
            idx=idx, path=frame_path, fc=frame_cnt,
            num_gt=len(gt), gt=gt_dump,
            num_prop=len(prop), prop=prop_dump
        ))

    open(out_list_name, 'w').writelines(processed_proposal_list)


def parse_directory(rgb_path, flow_path, key_func=lambda x: x[-15:-4],
                    rgb_prefix='image_', flow_x_prefix='flow_x_', flow_y_prefix='flow_y_'):
    """
    Parse directories holding extracted frames from standard benchmarks
    """
    print('parse frames under folder {}'.format(rgb_path))
    rgb_frame_folders = glob.glob(os.path.join(rgb_path, '*'))

    def count_files(rgb_directory, flow_directory, rgb_pre, flow_x_pre, flow_y_pre):
        if (not os.path.exists(rgb_directory)) or (not os.path.exists(flow_directory)):
            print('{} or {} is invalid!'.format(rgb_directory, flow_directory))
            return None
        rgb_lst = os.listdir(rgb_directory)
        flow_lst = os.listdir(flow_directory)
        cnt_list = list([])
        cnt_list.append(len(fnmatch.filter(rgb_lst, rgb_pre+'*')))
        cnt_list.append(len(fnmatch.filter(flow_lst, flow_x_pre + '*')))
        cnt_list.append(len(fnmatch.filter(flow_lst, flow_y_pre + '*')))
        return cnt_list

    # check RGB
    frame_dict = {}
    for i, f in enumerate(rgb_frame_folders):
        all_cnt = count_files(f, os.path.join(flow_path, f.split('/')[-1]), rgb_prefix, flow_x_prefix, flow_y_prefix)
        if all_cnt is None:
            continue
        k = key_func(f)

        x_cnt = all_cnt[1]
        y_cnt = all_cnt[2]
        if x_cnt != y_cnt:
            print('x and y direction have different number of flow images. video: '+f)
            continue
        if i % 200 == 0:
            print('{} videos parsed'.format(i))

        frame_dict[k] = (f, all_cnt[0], x_cnt)
    print('frame folder analysis done')
    return frame_dict

def dump_window_list(video_info, named_proposals, frame_path, name_pattern, allow_empty=True, score=None):

    # first count frame numbers
    try:
        video_name = video_info.path.split('/')[-1]
        files = glob.glob(os.path.join(frame_path, video_name, name_pattern))
        frame_cnt = len(files)
    except:
        if allow_empty:
            #frame_cnt = score.shape[0] * 6
            frame_cnt = 0
            video_name = video_info.id
        else:
            raise

    # convert time to frame number
    real_fps = float(frame_cnt) / float(video_info.duration)

    # get groundtruth windows
    gt_w = [(x.num_label, x.time_span) for x in video_info.instance]
    gt_windows = [(x[0]+1, int(x[1][0] * real_fps), int(x[1][1] * real_fps)) for x in gt_w]

    dump_gt = []
    for gt in gt_windows:
        dump_gt.append('{} {} {}'.format(*gt))

    dump_proposals = []
    for pr in named_proposals:
        real_start = int(pr[3] * real_fps)
        real_end = int(pr[4] * real_fps)
        label = pr[0]
        overlap = pr[1]
        overlap_self = pr[2]
        dump_proposals.append('{} {:.04f} {:.04f} {} {}'.format(label, overlap, overlap_self, real_start, real_end))

    ret_str = '{path}\n{duration}\n{fps}\n{num_gt}\n{gts}\n{num_window}\n{prs}\n'.format(
        path=os.path.join(frame_path, video_name), duration=frame_cnt, fps=1,
        num_gt=len(dump_gt), gts='\n'.join(dump_gt),
        num_window=len(dump_proposals), prs='\n'.join(dump_proposals))

    return ret_str
