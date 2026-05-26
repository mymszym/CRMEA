import copy
import json
import os
import random
from datetime import datetime
import pandas as pd
from tqdm import tqdm

from llm_classification import prompt_generation
from sim_pre_computation import generate_ptm_sim_dict_batch
from utility import read_groundtruth_with_mode


def evaluate_once_per_file(gt_dict: dict, model_output_dict: dict):
    num_hit_1 = 0
    num_error = 0
    num_null = 0
    num_non_hit = 0
    for key, value in model_output_dict.items():
        gt = gt_dict[key]
        if value == gt:
            num_hit_1 += 1
        elif value == 'error':
            num_error += 1
        elif value == 'null':
            num_null += 1
        else:
            num_non_hit += 1

    return num_hit_1, num_non_hit, num_error, num_null


def evaluate_rep_per_file(gt_dict: dict, model_output_dict: dict, k: int):
    rr = 0
    num_hit_1 = 0
    num_hit_k = 0
    num_non_hit = 0

    for ent1, ent2_pair_list in model_output_dict.items():
        gt = gt_dict[ent1]

        sorted_ent2_pair_list = sorted(ent2_pair_list, key=lambda x: x[1], reverse=True)
        top_k = [item[0] for item in sorted_ent2_pair_list[0:k]]

        rank = -1
        for index in range(len(top_k)):
            model_output = top_k[index]
            if model_output == gt:
                rank = index + 1
                break

        if rank == 1:
            num_hit_1 += 1
            num_hit_k += 1
            rr += (1/rank)
        elif rank == -1:
            num_non_hit += 1
        else:
            num_hit_k += 1
            rr += (1 / rank)

    return rr, num_hit_1, num_hit_k, num_non_hit


def evaluate(dataset_name: str, eval_mode: str):
    print('Evaluating ... dataset: %s with mode: %s' % (dataset_name, eval_mode))

    gt_dict = read_groundtruth_with_mode(dataset_name, 'all')
    if eval_mode == 'once':
        model_output_dir = os.path.join(os.getcwd(), '..', 'output', 'ea_result', dataset_name, 'processed')
    elif eval_mode == 'rep':
        model_output_dir = os.path.join(os.getcwd(), '..', 'output', 'ea_result', dataset_name, 'distribution')
    else:
        raise RuntimeError('Unknown eval_mode: %s' % eval_mode)

    eval_results_df = pd.DataFrame()
    for file_name in os.listdir(model_output_dir):
        suffix = os.path.splitext(file_name)[-1]
        prefix = file_name.split('=')[0]
        eval_tar = prefix.split('_')[0]
        # if prefix == 'test_distribution' and suffix == '.json':
        if suffix == '.json':
            fn = file_name[:-5]
            fn_splits = fn.split('=')
            llm_mode = fn_splits[1]
            ea_data_mode = fn_splits[2]
            triple_sel = fn_splits[3]
            if llm_mode == 'zs':
                exp_sel = ''
            else:
                exp_sel = fn_splits[4]
            tar_sel = fn_splits[5]
            k = int(tar_sel.split('-')[1])
            ptm_model = fn_splits[6]
            llm_type = fn_splits[7]
            llm_temp = fn_splits[8]
            if len(fn_splits) == 11:
                rep = 0
                lower_bound = fn_splits[9]
                upper_bound = fn_splits[10]
            elif len(fn_splits) == 12:
                rep = fn_splits[9]
                lower_bound = fn_splits[10]
                upper_bound = fn_splits[11]
            else:
                raise RuntimeError('model output format error: %s' % file_name)
            file_path = os.path.join(model_output_dir, file_name)
            with open(file_path, 'r', encoding='utf-8') as f:
                model_output_dict = json.load(f)
                num_hit_1 = -1
                num_hit_k = -1
                num_error = -1
                num_null = -1
                if eval_mode == 'once':
                    num_hit_1, num_non_hit, num_error, num_null = evaluate_once_per_file(gt_dict, model_output_dict)
                elif eval_mode == 'rep':
                    rr, num_hit_1, num_hit_k, num_non_hit = evaluate_rep_per_file(gt_dict, model_output_dict, k)
                else:
                    raise RuntimeError('Unknown eval_mode: %s' % eval_mode)
                num_total = len(model_output_dict)
                hit_1 = num_hit_1 / num_total
                hit_k = num_hit_k / num_total
                mrr = rr / num_total
                if eval_mode == 'once':
                    llm_accuracy = -1
                elif eval_mode == 'rep':
                    if hit_k == 0:
                        llm_accuracy = 0
                    else:
                        llm_accuracy = hit_1 / hit_k
                else:
                    raise RuntimeError('Unknown eval_mode: %s' % eval_mode)

                cur_eval_result_dict = {
                    'file_name': file_name, 'eval_tar': eval_tar, 'hit_1': hit_1, 'hit_k': hit_k, 'LA': llm_accuracy,
                    'mrr': mrr, 'triple_sel': triple_sel, 'ptm_model': ptm_model, 'llm_mode': llm_mode,
                    'tar_sel': tar_sel, 'lower_bound': lower_bound, 'upper_bound': upper_bound,
                    'dataset_name': dataset_name, 'ea_data_mode': ea_data_mode, 'exp_sel': exp_sel,
                    'llm_type': llm_type, 'llm_temp': llm_temp, 'rep': rep,
                    'num_total': num_total, 'num_hit_1': num_hit_1, 'num_hit_k': num_hit_k,
                    'num_non_hit': num_non_hit, 'num_error': num_error, 'num_null': num_null
                }
                cur_eval_result_df = pd.DataFrame(cur_eval_result_dict, index=[0])
                eval_results_df = pd.concat([eval_results_df, cur_eval_result_df])

    now = datetime.now()
    eval_results_dir = os.path.join(os.getcwd(), '..', 'evaluation', dataset_name)
    if not os.path.exists(eval_results_dir):
        os.makedirs(eval_results_dir)
    eval_results_path = os.path.join(eval_results_dir, 'eval_result_%s_m%d-d%d-h%d-m%d.xlsx' %
                                     (eval_mode, now.month, now.day, now.hour, now.minute))
    eval_results_df.to_excel(eval_results_path, index=False)
    print('File generated successfully...%s' % eval_results_path)

    return 0


def get_gt_index_and_value(llm_top_5: list, slm_top_5: list, gt: str):
    gt_rank_llm = -1
    gt_hit_llm = -1
    gt_rank_slm = -1

    for index in range(len(llm_top_5)):
        ent_hit_pair = llm_top_5[index]
        ent = ent_hit_pair[0]
        if ent == gt:
            gt_rank_llm = index + 1
            gt_hit_llm = int(ent_hit_pair[1])
            break

    for index in range(len(slm_top_5)):
        ent = slm_top_5[index]
        if ent == gt:
            gt_rank_slm = index + 1
            break

    return gt_rank_llm, gt_hit_llm, gt_rank_slm


def generate_error_case_analysis_rep_per_file(gt_dict: dict, model_output_dict: dict,
                                              dataset_name: str, file_name: str):
    fn = file_name[:-5]
    fn_splits = fn.split('=')
    llm_mode = fn_splits[1]
    ea_data_mode = fn_splits[2]
    triple_sel = fn_splits[3]
    if llm_mode == 'zs':
        exp_sel = ''
    else:
        exp_sel = fn_splits[4]

    llm_error_case_df = pd.DataFrame()
    slm_error_case_df = pd.DataFrame()
    for ent1, ent2_pair_list in tqdm(model_output_dict.items()):
        slm_top_5 = [item[0] for item in ent2_pair_list[0:5]]
        sorted_ent2_pair_list = sorted(ent2_pair_list, key=lambda x: x[1], reverse=True)
        llm_top_5 = sorted_ent2_pair_list[0:5]
        gt = gt_dict[ent1]
        gt_rank_llm, gt_hit_llm, gt_rank_slm = get_gt_index_and_value(llm_top_5, slm_top_5, gt)

        ent2_list = []
        for ent2_pair in ent2_pair_list:
            ent2 = ent2_pair[0]
            ent2_list.append(ent2)
        prompt = prompt_generation(dataset_name, llm_mode, ea_data_mode, triple_sel,
                                   exp_sel, ent1, ent2_list)
        cur_slm_error_dict = {'ent': ent1, 'ground_truth': gt, 'slm_output': str(ent2_list),
                              'gt_rank_slm': gt_rank_slm, 'prompt': prompt, 'remark': ''}
        cur_slm_error_df = pd.DataFrame(cur_slm_error_dict, index=[0])
        slm_error_case_df = pd.concat([slm_error_case_df, cur_slm_error_df])

        if gt_rank_llm > 1:
            cur_llm_error_dict = {'ent': ent1, 'ground_truth': gt, 'llm_output': str(sorted_ent2_pair_list),
                                  'gt_rank_llm': gt_rank_llm, 'gt_hit_llm': gt_hit_llm, 'gt_rank_slm': gt_rank_slm,
                                  'prompt': prompt, 'remark': ''}
            cur_df = pd.DataFrame(cur_llm_error_dict, index=[0])
            llm_error_case_df = pd.concat([llm_error_case_df, cur_df])

    ## save slm error case as file
    slm_error_case_dir = os.path.join(os.getcwd(), '..', 'evaluation', dataset_name, 'slm_error_case')
    if not os.path.exists(slm_error_case_dir):
        os.makedirs(slm_error_case_dir)
    slm_error_case_path = os.path.join(slm_error_case_dir, fn + '.xlsx')
    slm_error_case_df.to_excel(slm_error_case_path, index=False)
    print('File generated successfully...%s' % slm_error_case_path)
    ## save llm error case as file
    llm_error_case_dir = os.path.join(os.getcwd(), '..', 'evaluation', dataset_name, 'llm_error_case')
    if not os.path.exists(llm_error_case_dir):
        os.makedirs(llm_error_case_dir)
    llm_error_case_path = os.path.join(llm_error_case_dir, fn + '.xlsx')
    llm_error_case_df.to_excel(llm_error_case_path, index=False)
    print('File generated successfully...%s' % llm_error_case_path)

    return 0


def error_case_analysis(dataset_name: str, eval_mode: str):
    print('Analyzing LLM error cases ... dataset: %s with mode: %s' % (dataset_name, eval_mode))

    gt_dict = read_groundtruth_with_mode(dataset_name, 'all')
    if eval_mode == 'once':
        model_output_dir = os.path.join(os.getcwd(), '..', 'output', 'ea_result', dataset_name, 'processed')
    elif eval_mode == 'rep':
        model_output_dir = os.path.join(os.getcwd(), '..', 'output', 'ea_result', dataset_name, 'distribution')
    else:
        raise RuntimeError('Unknown eval_mode: %s' % eval_mode)

    for file_name in os.listdir(model_output_dir):
        prefix = 'test_distribution=zs=train-20=rand-5'
        suffix = 'mpnet-round-2=gpt3.5=0=5=0=2700.json'
        if file_name.startswith(prefix) and file_name.endswith(suffix):
            file_path = os.path.join(model_output_dir, file_name)
            with open(file_path, 'r', encoding='utf-8') as f:
                model_output_dict = json.load(f)
                generate_error_case_analysis_rep_per_file(gt_dict, model_output_dict, dataset_name, file_name)

    return 0


if __name__ == '__main__':
    ## ['DBP15K_FR_EN', 'DBP15K_JA_EN', 'DBP15K_ZH_EN', 'DW15K_V1', 'DY15K_V1',
    # 'DBP15K_DE_EN_V1', 'DBP15K_FR_EN_V1', 'DBP100K_DE_EN_V1', 'DBP100K_FR_EN_V1']
    dataset_name = 'DBP15K_FR_EN_V1'
    ## rep or once
    eval_mode = 'rep'
    evaluate(dataset_name, eval_mode)
    # error_case_analysis(dataset_name, eval_mode)
