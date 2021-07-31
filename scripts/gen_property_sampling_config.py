import json
import os
import itertools
import shutil
from typing import ValuesView

local_samples = 16


class cPropertyManager:
    def __init__(self):
        self.property_dict = {}
        self.property_sample_lst = []

    def register(self, prop_name, prop_id):
        assert (prop_name in self.property_dict) == False
        assert prop_id == len(
            self.property_dict
        ), f"please add key in index ascending order {prop_id}"
        self.property_dict[prop_name] = prop_id

    def show_property_dict(self):
        print(self.property_dict)

    def dump_to_json(self, path):
        self._dump_to_json_segment(path, 0, len(self.property_sample_lst))

    def _dump_to_json_segment(self, path, idx_st, idx_ed):
        prop_name_list = [i for i in self.property_dict]
        value = {
            "prop_name_list": prop_name_list,
            "prop_samples": self.property_sample_lst[idx_st:idx_ed],
            "external_property_samples_start_id": idx_st * local_samples
        }
        dirname = os.path.dirname(path)
        if os.path.exists(dirname) == False:
            os.makedirs(dirname)
            print(f"create {dirname}")
        with open(path, 'w') as f:
            json.dump(value, f, indent=True)
        # print(f"write down to {path}")

    def dump_to_multiple_path(self, path_lst, weight_lst):
        num_path = len(path_lst)
        assert num_path >= 1
        segment_lst = []
        cur_st = 0
        num_total_sample = len(self.property_sample_lst)
        standard_gap = round(num_total_sample / num_path)
        print(f"total sample {num_total_sample}, num_machine {num_path}, gap {standard_gap}")
        assert num_total_sample >= num_path
        for i in range(num_path):
            if i == num_path - 1:
                segment_lst.append((cur_st, num_total_sample))
            else:
                cur_ed = cur_st + round(standard_gap * weight_lst[i])
                segment_lst.append((cur_st, cur_ed))
                cur_st = cur_ed

        # print(f"sample num {num_total_sample}, seg {segment_lst}")
        for i in range(num_path):
            path = path_lst[i]
            st = segment_lst[i][0]
            ed = segment_lst[i][1]
            self._dump_to_json_segment(path, st, ed)
            print(
                f"[log] dump from {st} to {ed} to {path}, samples {ed - st + 1}, weight {weight_lst[i]}"
            )

    def add(self, prop):
        assert len(prop) == len(self.property_dict)
        self.property_sample_lst.append(prop)


def gen_isotropic_config(mana, path):
    bending_begin = 0
    bending_end = 50
    samples = 200
    stretch = 27
    for i in range(samples):
        cur_bending = bending_begin + (bending_end -
                                       bending_begin) / (samples - 1) * i
        mana.add(
            [stretch, stretch, stretch, cur_bending, cur_bending, cur_bending])

    mana.dump_to_json()


from copy import deepcopy


def remove_duplicate(value_lst):
    old_lst = deepcopy(value_lst)

    def find_and_delete(lst, raw_value):
        new_value = deepcopy(raw_value)
        new_value[0] = raw_value[2]
        new_value[2] = raw_value[0]
        delete = False

        assert (raw_value in lst) or (new_value in lst)
        if raw_value in lst and new_value in lst:
            # there may delete two elements
            idx = old_lst.index(raw_value)
            del old_lst[idx]
            delete = True
            if (raw_value in lst) == False and (new_value in lst) == False:
                assert new_value == old_value
                lst.insert(idx, new_value)
                # lst.append(new_value)
                delete = False

        assert (raw_value in lst) or (
            new_value in lst), f"new value {new_value}, old_value {raw_value}"
        return delete, lst

    count = 0
    for old_value in value_lst:
        delete, old_lst = find_and_delete(old_lst, old_value)
        if delete:
            count += 1

    print(f"raw len {len(value_lst)}, new len {len(old_lst)}, deleted {count}")
    return old_lst


def create_data_synthesis_config(template_path, new_sample_lst_path,
                                 output_batch_config_path):
    assert os.path.exists(template_path)
    shutil.copyfile(template_path, output_batch_config_path)
    import json

    sample_name = os.path.split(new_sample_lst_path)[-1]
    output_dir = sample_name[:sample_name.find(".")]
    # print(f"sample name {sample_name}")
    # print(f"output dir  {output_dir}")

    with open(output_batch_config_path, 'r') as f:
        cont = json.load(f)
        cont["property_manager"]["enable_external_property_samples"] = True
        cont["property_manager"][
            "external_property_samples_path"] = new_sample_lst_path
        cont["export_data_dir"] = output_dir

    with open(output_batch_config_path, 'w') as f:
        json.dump(cont, f, indent=4)

    print(
        f"[log] create new data synthesis config {output_batch_config_path} succ"
    )


def create_bat(bat_id, config_path):
    bat_name = f"../run{bat_id}.bat"
    with open(bat_name, 'w') as f:
        f.writelines(f"main.exe {config_path}")
    print(f"[log] create batch file {bat_name} succ")


def gen_uniform_config(manager, path_lst, weight_lst):
    samples = 25
    bending_begin = 1
    bending_end = 50
    stretch = 27
    gap = samples - 1
    step = (bending_end - bending_begin) / gap if gap != 0 else 0

    print(f"begin st {bending_begin} begin ed {bending_end} step {step}")
    uniform_values = [bending_begin + step * i for i in range(samples)]

    values = [
        list(i) for i in itertools.product(uniform_values, uniform_values,
                                           uniform_values)
    ]
    values = remove_duplicate(values)
    for cur_v in values:
        manager.add([stretch, stretch, stretch, cur_v[0], cur_v[1], cur_v[2]])
    manager.dump_to_multiple_path(path_lst, weight_lst)

    print(f"-----------------------")
    template_data_synthesis_config = "../config/data_synthesis.json"

    for _idx, cur_path in enumerate(path_lst):
        # 1. copy and generate the export data dir and path
        new_config_path = f"split_save/batch_data_synthesis_part{_idx}.json"

        # 2. create the bat file, copy the files to the main directory
        create_data_synthesis_config(template_data_synthesis_config,
                                     os.path.join("scripts", cur_path),
                                     new_config_path)
        create_bat(_idx, os.path.join("scripts", new_config_path))


if __name__ == "__main__":
    mana = cPropertyManager()
    mana.register("stretch_warp", 0)
    mana.register("stretch_weft", 1)
    mana.register("stretch_bias", 2)
    mana.register("bending_warp", 3)
    mana.register("bending_weft", 4)
    mana.register("bending_bias", 5)

    # for isotropi config
    # path = "../config/train_configs/isotropic_properties_sample_200.json"
    # gen_isotropic_config()

    num = 22
    # :
    path_lst = [ f"split_save/uniform_3c_sample25_noised16_amp5e-4_xgpu_part{i}.json" for i in range(num)]
    # path_lst = [
    #     "split_save/uniform_3c_sample25_noised16_amp5e-4_xgpu_part0.json",
    #     "split_save/uniform_3c_sample25_noised16_amp5e-4_xgpu_part1.json",
    #     "split_save/uniform_3c_sample25_noised16_amp5e-4_xgpu_part2.json",
    #     "split_save/uniform_3c_sample25_noised16_amp5e-4_xgpu_part3.json",
    #     "split_save/uniform_3c_sample25_noised16_amp5e-4_xgpu_part4.json",
    #     "split_save/uniform_3c_sample25_noised16_amp5e-4_xgpu_part5.json",
    #     "split_save/uniform_3c_sample25_noised16_amp5e-4_xgpu_part6.json",
    #     "split_save/uniform_3c_sample25_noised16_amp5e-4_xgpu_part7.json",
    #     "split_save/uniform_3c_sample25_noised16_amp5e-4_xgpu_part8.json",
    # ]
    import numpy as np
    # weight_lst = [0.87,0.7,0.77,0.92,1.14,1.36,0.70,1.22,0.70]
    weight_lst = [ 1 for _ in range(num)]
    # weight_lst /= (np.sum(weight_lst) / len(weight_lst))
    gen_uniform_config(mana, path_lst, weight_lst)
    # gen_two_channels_uniform_config(mana, path_lst)

# def gen_two_channels_uniform_config(manager, path_lst):
#     samples = 13
#     st_val, ed_val = 1, 50
#     stretch_val = 27
#     gap = samples - 1

#     template_data_synthesis_config = "../config/data_synthesis.json"

#     step = (ed_val - st_val) / gap if gap != 0 else 0
#     single_value_lst = [25]

#     uniform_values = [st_val + step * i for i in range(samples)]
#     values = [
#         list(i) for i in itertools.product(single_value_lst, uniform_values,
#                                            uniform_values)
#     ]
#     values = remove_duplicate(values)
#     for cur_v in values:
#         manager.add([
#             stretch_val, stretch_val, stretch_val, cur_v[0], cur_v[1], cur_v[2]
#         ])
#     manager.dump_to_multiple_path(path_lst)

#     print(f"-----------------------")

#     for _idx, cur_path in enumerate(path_lst):
#         # 1. copy and generate the export data dir and path
#         # 2. create the bat file, copy the files to the main directory

#         new_config_path = f"split_save/batch_data_synthesis_part{_idx}.json"
#         # print(
#         #     f"old config {template_data_synthesis_config} new config {new_config_path}"
#         # )
#         create_data_synthesis_config(template_data_synthesis_config, os.path.join("scripts", cur_path),
#                                      new_config_path)
#         create_bat(_idx, os.path.join("scripts",  new_config_path))
#     # generate batch files and corresponding config