import json
from copy import deepcopy
from .filter import Filter


class DeleteSameFilter(Filter):
    """
    For QA data, there may be some redundant data, which need to be filtered.
    This version temporarily stores image data during comparison and restores it afterwards.
    """

    @classmethod
    def do_filter(cls, data):
        unique_data = set()
        delete_same_output = []
        temp_image_store = {}

        for idx, obj in enumerate(data):
            obj_copy = deepcopy(obj)

            image_data = obj_copy.pop('image')
            temp_key = f"temp_key_{idx}"
            temp_image_store[temp_key] = (image_data, obj_copy)

            data_str = json.dumps(obj_copy, sort_keys=True)

            if data_str not in unique_data:
                unique_data.add(data_str)
                if temp_key in temp_image_store:
                    stored_image, _ = temp_image_store[temp_key]
                    obj_copy['image'] = stored_image
                delete_same_output.append(obj_copy)

        return delete_same_output
