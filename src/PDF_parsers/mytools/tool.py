import re


def is_chinese(word):
    """
    判断一个字符串是否为中文
    :param word: 字符串
    :return: True or False
    """
    pattern = "^[\u4e00-\u9fa5]+$"
    if re.match(pattern, word):
        return True
    else:
        return False