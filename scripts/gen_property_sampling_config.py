import json


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
            "prop_samples": self.property_sample_lst[idx_st:idx_ed]
        }
        with open(path, 'w') as f:
            json.dump(value, f)
        print(f"write down to {path}")

    def dump_to_multiple_path(self, path_lst):
        num_path = len(path_lst)
        assert num_path >= 1
        segment_lst = []
        cur_st = 0
        num_total_sample = len(self.property_sample_lst)
        assert num_total_sample >= num_path
        for i in range(num_path):
            if i == num_path - 1:
                segment_lst.append((cur_st, num_total_sample))
            else:
                cur_ed = cur_st + num_total_sample // num_path
                segment_lst.append((cur_st, cur_ed))
                cur_st = cur_ed
        print(f"sample num {num_total_sample}, seg {segment_lst}")
        for i in range(num_path):
            path = path_lst[i]
            st = segment_lst[i][0]
            ed = segment_lst[i][1]
            self._dump_to_json_segment(path, st, ed)
            print(f"dump from {st} to {ed} to {path}")
        exit()

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


def gen_uniform_config(manager, path_lst):
    samples = 10
    bending_begin = 1
    bending_end = 50
    stretch = 27
    gap = samples - 1
    step = (bending_end - bending_begin) / gap if gap != 0 else 0

    print(f"begin st {bending_begin} begin ed {bending_end} step {step}")
    uniform_values = [bending_begin + step * i for i in range(samples)]

    import itertools
    values = [
        list(i) for i in itertools.product(uniform_values, uniform_values,
                                           uniform_values)
    ]
    values = remove_duplicate(values)
    for cur_v in values:
        manager.add([stretch, stretch, stretch, cur_v[0], cur_v[1], cur_v[2]])
    manager.dump_to_multiple_path(path_lst)
    # for i in manager.property_sample_lst:
    #     print(i)


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

    path_lst = [
        "split_save/uniform_sample10_noised16_amp1e-4_part0.json",
        "split_save/uniform_sample10_noised16_amp1e-4_part1.json",
        "split_save/uniform_sample10_noised16_amp1e-4_part2.json",
        "split_save/uniform_sample10_noised16_amp1e-4_part3.json",
        "split_save/uniform_sample10_noised16_amp1e-4_part4.json"
    ]

    gen_uniform_config(mana, path_lst)