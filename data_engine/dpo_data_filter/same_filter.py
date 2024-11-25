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
        temp_image_store = {}  # 用于暂存 image 数据

        for idx, obj in enumerate(data):
            # 创建对象的深拷贝
            obj_copy = deepcopy(obj)

            # 如果存在 image 字段，暂存它
            if 'image' in obj_copy:
                # 使用对象的其他字段作为键来存储 image
                image_data = obj_copy.pop('image')
                temp_key = f"temp_key_{idx}"  # 使用索引创建唯一的临时键
                temp_image_store[temp_key] = (image_data, obj_copy)

            # 将处理后的数据序列化为字符串
            data_str = json.dumps(obj_copy, sort_keys=True)

            if data_str not in unique_data:
                unique_data.add(data_str)
                # 如果有 image，从临时存储中恢复它
                if temp_key in temp_image_store:
                    stored_image, _ = temp_image_store[temp_key]
                    obj_copy['image'] = stored_image
                delete_same_output.append(obj_copy)

        return delete_same_output
