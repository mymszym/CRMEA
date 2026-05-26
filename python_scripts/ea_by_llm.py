import copy
import json
import os
import random
import time

import numpy as np
from tqdm import tqdm

from llm_classification import llm_classify
from sim_pre_computation import generate_ptm_sim_dict_batch
from utility import read_groundtruth_with_mode


def select_top_scores(data, std_fraction=10, max_recall=5):
    """
    Select top scores from each list in the dictionary.

    Parameters:
    - data: A dictionary with items as keys and lists of tuples as values. Each tuple contains an object and a score.
    - std_fraction: The fraction of the standard deviation to add to the mean gap to determine the threshold for significant gaps.

    Returns:
    - A dictionary with the same keys as `data` and lists of selected tuples as values.
    """
    selected_results = {}

    for key, values in data.items():
        # Extracting scores
        scores = [item[1] for item in values]

        # Calculating gaps between consecutive scores
        gaps = [scores[i] - scores[i + 1] for i in range(len(scores) - 1)]

        # Calculate mean and adjusted standard deviation of gaps
        mean_gap = np.mean(gaps)
        adjusted_std_gap = std_fraction * np.std(gaps)
        significant_gap = mean_gap + adjusted_std_gap

        # Find the index of the first significant gap
        cutoff_index = next((i for i, gap in enumerate(gaps) if gap > significant_gap), len(scores))

        # Select the items above the significant gap
        selected_items = values[:cutoff_index + 1][:max_recall]
        selected_results[key] = selected_items

    #print average recall
    total = 0
    recall_nums = []
    for key, values in selected_results.items():
        total += len(values)
        recall_nums.append(len(values))
    print(f'average recall: {total/len(selected_results)}')

    return selected_results


def ea_by_llm(dataset_name: str, llm_mode: str, ea_data_mode: str, triple_sel: str, exp_sel: str, tar_sel: str,
              ptm_model: str, llm_type: str, llm_model: str, llm_temp: str,
              rep: int, lower_bound: int, upper_bound: int, train_percentage: float, max_rep: int):
    gt_dict = read_groundtruth_with_mode(dataset_name, 'all')
    ent1_list = list(gt_dict.keys())

    ea_dict = {}
    ptm_sim_dict = {}
    tar_ent1_list = ent1_list[lower_bound:upper_bound]
    tar_splits = tar_sel.split('-')
    num_class = int(tar_splits[1])
    triple_splits = triple_sel.split('-')
    triple_strategy = triple_splits[0]
    if len(tar_splits) == 2:
        tar_rank = 'top'
    else:
        tar_rank = 'sg'

    ## load ptm_sim_dict if tar_strategy is 'ptm'
    if triple_strategy == 'ptm':
        if '15K' in dataset_name:
            batch_size = 300
        elif '100K' in dataset_name:
            batch_size = 125
        else:
            raise RuntimeError('Unhandled dataset_name: %s for ea_by_llm' % dataset_name)
        ## check if debugging mode
        if upper_bound == 100:
            ptm_sim_upper_bound = 3000
        else:
            ptm_sim_upper_bound = upper_bound

        if ea_data_mode == 'train-20':
            train_bound = int(ptm_sim_upper_bound * train_percentage)
        elif ea_data_mode == 'test-80':
            train_bound = int(lower_bound * train_percentage)
        else:
            raise RuntimeError('Unhandled ea_data_mode: %s for ea_by_llm' % ea_data_mode)

        ptm_sim_dict = generate_ptm_sim_dict_batch(dataset_name, ea_data_mode, ptm_model, batch_size, llm_mode,
                                                   triple_sel, exp_sel, tar_sel, llm_type, llm_temp,
                                                   max_rep, lower_bound, train_bound, ptm_sim_upper_bound)
        if tar_rank == 'sg':
            std_fraction = 10
            for ent1 in ptm_sim_dict.keys():
                ent2_sim_pair_list = ptm_sim_dict[ent1]
                tar_ent2_sim_pair_list = sorted(ent2_sim_pair_list, key=lambda x: x[1], reverse=True)
                ptm_sim_dict[ent1] = tar_ent2_sim_pair_list
            ptm_sim_dict = select_top_scores(ptm_sim_dict, std_fraction, num_class)

    cur_index = lower_bound
    for ent1 in tqdm(tar_ent1_list):
        cur_index += 1
        ## retrieve target ent2 list
        if triple_strategy == 'ptm':
            tar_ent2_list = []
            if tar_rank == 'sg':
                tar_ent2_sim_pair_list = ptm_sim_dict[ent1]
            else:
                ent2_sim_pair_list = ptm_sim_dict[ent1]
                # tar_ent2_sim_pair_list = sorted(ent2_sim_pair_list, key=lambda x: x[1], reverse=True)
                tar_ent2_sim_pair_list = ent2_sim_pair_list[:num_class]
            for ent2_sim_pair in tar_ent2_sim_pair_list:
                tar_ent2_list.append(ent2_sim_pair[0])
        elif triple_strategy == 'rand':
            gt_ent2 = gt_dict[ent1]
            temp_gt_dict = copy.deepcopy(gt_dict)
            del temp_gt_dict[ent1]
            tar_ent2_list = random.choices(list(temp_gt_dict.values()), k=(num_class - 1))
            tar_ent2_list.append(gt_ent2)
        else:
            raise RuntimeError('Unknown triple selection strategy: %s' % triple_strategy)
        ## retrieve llm_aligned_ent2
        if len(tar_ent2_list) > 1:
            llm_aligned_ent2 = ''
            try:
                llm_aligned_ent2 = llm_classify(dataset_name, llm_mode, ea_data_mode, triple_sel, exp_sel,
                                                ent1, tar_ent2_list, llm_type, llm_model, llm_temp)
                if isinstance(llm_aligned_ent2, str):
                    if llm_aligned_ent2.startswith('[') and llm_aligned_ent2.endswith(']'):
                        llm_aligned_ent2 = llm_aligned_ent2[1:-1]
                else:
                    print('LLM return null for entity: %s' % ent1)
                    llm_aligned_ent2 = 'null'
            except:
                print('LLM classification error for entity: %s' % ent1)
                llm_aligned_ent2 = 'error'
            finally:
                ea_dict[ent1] = llm_aligned_ent2
        else:
            llm_aligned_ent2 = tar_ent2_list[0]
            ea_dict[ent1] = llm_aligned_ent2

    if cur_index > lower_bound:
        ea_output_dir = os.path.join(os.getcwd(), '..', 'output', 'ea_result', dataset_name)
        if not os.path.exists(ea_output_dir):
            os.makedirs(ea_output_dir)
        ea_output_file_name = '%s=%s=%s=%s=%s=%s=%s=%s=%d=%d=%d.json' % \
                              (llm_mode, ea_data_mode, tar_sel, exp_sel, triple_sel,
                               ptm_model, llm_type, llm_temp, rep, lower_bound, cur_index)
        ea_output_file_path = os.path.join(ea_output_dir, ea_output_file_name)
        with open(ea_output_file_path, 'w', encoding='utf-8') as f:
            json.dump(ea_dict, f, ensure_ascii=False, indent=4)
        print('Output generated successfully: %s' % ea_output_file_path)

    return 0


def get_current_upper_bound(dataset_name: str, llm_mode: str, ea_data_mode: str, triple_sel: str,
                            exp_sel: str, tar_sel: str, ptm_model: str, llm_type: str, llm_temp: str, repetition: int):
    upper_bound = 0
    model_output_dir = os.path.join(os.getcwd(), '..', 'output', 'ea_result', dataset_name)
    if not os.path.exists(model_output_dir):
        os.makedirs(model_output_dir)
    filenames = os.listdir(model_output_dir)
    if len(filenames) == 0:
        return 0
    else:
        prefix = ('%s=%s=%s=%s=%s=%s=%s=%s=%d' %
                  (llm_mode, ea_data_mode, tar_sel, exp_sel, triple_sel,
                   ptm_model, llm_type, llm_temp, repetition))
        for file_name in filenames:
            suffix = os.path.splitext(file_name)[-1]
            if file_name.startswith(prefix) and suffix == '.json':
                base_name = os.path.splitext(file_name)[0]
                split_list = base_name.split('=')
                cur_upper = int(split_list[-1])
                if upper_bound < cur_upper:
                    upper_bound = cur_upper

    return upper_bound


def ea_by_llm_recursively(dataset_name: str, llm_mode: str, ea_data_mode: str, triple_sel: str,
                          exp_sel: str, tar_sel: str, ptm_model: str, llm_type: str, llm_model: str, llm_temp: str,
                          max_rep: int, lower_bound: int, upper_bound: int, train_percentage: float):
    for rep in range(0, max_rep):
        print("Processing repetition %d/%d" % (rep + 1, max_rep))
        cur_upper = get_current_upper_bound(dataset_name, llm_mode, ea_data_mode, triple_sel, exp_sel, tar_sel,
                                            ptm_model, llm_type, llm_temp, rep)
        while cur_upper < upper_bound:
            if cur_upper > lower_bound:
                lower_bound = cur_upper

            print('LLM annotating... %s=%s=%s=%s=%s=%s=%s=%s=%s=%d=%d=%d.json' %
                  (dataset_name, llm_mode, ea_data_mode, triple_sel, exp_sel, tar_sel,
                   ptm_model, llm_type, llm_temp, rep, lower_bound, upper_bound)
                  )
            ea_by_llm(dataset_name, llm_mode, ea_data_mode, triple_sel, exp_sel, tar_sel, ptm_model,
                      llm_type, llm_model, llm_temp, rep, lower_bound, upper_bound, train_percentage, max_rep)

            cur_upper = get_current_upper_bound(dataset_name, llm_mode, ea_data_mode, triple_sel, exp_sel, tar_sel,
                                                ptm_model, llm_type, llm_temp, rep)
            time.sleep(1)

    return 0


def complete_llm_output(dataset_name: str, llm_mode: str, ea_data_mode: str, num_triple: int, exp_sel: str,
                        tar_sel: str, ptm_model: str, llm_type: str, llm_model: str, llm_temp: str, rep: int,
                        lower_bound: int, upper_bound: int, train_percentage: float):
    train_bound = int(upper_bound * train_percentage)
    output_dir = os.path.join(os.getcwd(), '..', 'output', 'ea_result', dataset_name)
    output_file_name = '%s=%s=%d=%s=%s=%s=%s=%s=%d=%d=%d.json' % \
                       (llm_mode, ea_data_mode, num_triple, exp_sel, tar_sel,
                        ptm_model, llm_type, llm_temp, rep, lower_bound, upper_bound)
    output_file_path = os.path.join(output_dir, output_file_name)
    with open(output_file_path, 'r', encoding='utf-8') as f:
        output_dict = json.load(f)

    ptm_sim_dict = {}
    gt_dict = read_groundtruth_with_mode(dataset_name, ea_data_mode)
    tar_splits = tar_sel.split('-')
    tar_strategy = tar_splits[0]
    num_class = int(tar_splits[1])
    element = tar_splits[2]
    desc_model = tar_splits[3]
    if tar_strategy == 'ptm':
        batch_size = 300
        generate_ptm_sim_dict_batch(dataset_name, ea_data_mode, ptm_model, batch_size, llm_mode, triple_sel,
                                    exp_sel, tar_sel, llm_type, llm_temp, max_rep,
                                    lower_bound, train_bound, upper_bound)

    num_error = 0
    num_null = 0
    num_completion = 0
    for ent1, ent2 in tqdm(output_dict.items()):
        if ent2 in ['error', 'null']:
            if ent2 == 'error':
                num_error += 1
            else:
                num_null += 1

            if tar_strategy == 'ptm':
                ent2_sim_pair_list = ptm_sim_dict[ent1]
                sorted_ent2_sim_pair_list = sorted(ent2_sim_pair_list, key=lambda x: x[1], reverse=True)
                tar_ent2_list = []
                for ent2_sim_pair in sorted_ent2_sim_pair_list[:num_class]:
                    tar_ent2_list.append(ent2_sim_pair[0])
            elif tar_strategy == 'rand':
                gt_ent2 = gt_dict[ent1]
                temp_gt_dict = copy.deepcopy(gt_dict)
                del temp_gt_dict[ent1]
                tar_ent2_list = random.choices(list(temp_gt_dict.values()), k=(num_class - 1))
                tar_ent2_list.append(gt_ent2)
            else:
                raise RuntimeError('Unknown target selection strategy: %s' % tar_sel)

            llm_aligned_ent2 = ''
            try:
                llm_aligned_ent2 = llm_classify(dataset_name, llm_mode, ea_data_mode, triple_sel, exp_sel,
                                                ent1, tar_ent2_list, llm_type, llm_model, llm_temp)
                num_completion += 1
                if not isinstance(llm_aligned_ent2, str):
                    print('LLM return null for entity: %s' % ent1)
                    llm_aligned_ent2 = 'null'
                    num_completion -= 1
            except:
                print('LLM classification error for entity: %s' % ent1)
                llm_aligned_ent2 = 'error'
                num_completion -= 1
            finally:
                output_dict[ent1] = llm_aligned_ent2

    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(output_dict, f, ensure_ascii=False, indent=4)
    print('Output generated successfully: %s' % output_file_path)
    print('Completion status: error (%d), null (%d), completion (%d)' % (num_error, num_null, num_completion))

    return 0


if __name__ == '__main__':
    dataset_name = 'DW15K_V1'
    ea_data_mode = 'train-20'
    ptm_model = 'labse'
    ## gpt3.5, gpt4, baichuan2-v100, baichuan2-a6000
    llm_type = 'gpt3.5'
    llm_model = 'gpt-3.5-turbo-1106'
    llm_temp = '0'
    # range of test cases
    lower_bound = 0
    upper_bound = 3000
    max_rep = 5
    train_percentage = 0.9

    llm_mode = 'icl'
    exp_sel = ''
    tar_sel = 'ptm-5-triple-freq'
    triple_sel = 'freq-5'
    ea_by_llm_recursively(
        dataset_name, llm_mode, ea_data_mode, triple_sel, exp_sel, tar_sel,
        ptm_model, llm_type, llm_model, llm_temp, max_rep,
        lower_bound, upper_bound, train_percentage
    )
