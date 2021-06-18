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
        prop_name_list = [i for i in self.property_dict]
        value = {
            "prop_name_list" : prop_name_list,
            "prop_samples" : self.property_sample_lst
        }
        with open(path, 'w') as f:
            json.dump(value, f)
        print(f"write down to {path}")

    def add(self, prop):
        assert len(prop) == len(self.property_dict)
        self.property_sample_lst.append(prop)


if __name__ == "__main__":
    mana = cPropertyManager()
    mana.register("stretch_warp", 0)
    mana.register("stretch_weft", 1)
    mana.register("bending_warp", 2)
    mana.register("bending_weft", 3)
    mana.register("bending_bias", 4)

    bending_begin = 5e2
    bending_end = 5e3
    samples = 200
    stretch = 1e5
    for i in range(samples):
        cur_bending = bending_begin + (bending_end -
                                       bending_begin) / (samples - 1) * i
        mana.add([stretch, stretch, cur_bending, cur_bending, cur_bending])

    mana.dump_to_json("../config/train_configs/isotropic_properties_sample_200.json")