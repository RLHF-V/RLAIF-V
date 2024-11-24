import json

from .filter import Filter


class DeleteSameFilter(Filter):
    """
    For QA data, there may be some redundant data, which need to be filtered
    """

    @classmethod
    def do_filter(cls, data):
        unique_data = set()
        delete_same_output = []
        for obj in data:
            # 将每条数据序列化为字符串，便于使用集合去重
            data_str = json.dumps(obj, sort_keys=True)
            if data_str not in unique_data:
                unique_data.add(data_str)
                delete_same_output.append(obj)

        # print(f"去重完成，共写入 {len(unique_data)} 条数据到 {output_file}")

        return delete_same_output
