'''
-*- coding: utf-8 -*-
@File  : status.py.py
@author: wangxiang4@tal
@Time  : 2023/09/20 20:59
'''
from enum import Enum


class Status(Enum):
    ######### 2000X: tonado模块报错 ################
    SUCCESS = {20000: 'Success.'}  # 正常编码
    PARAMETER_ERROR = {20001: 'Parsing error.'}  # 入参解析错误
    TIME_OUT = {20002: 'Timeout.'}  # 超时
    TASK_ABORT = {20003: 'Task abort.'}  # 调用接口失败
    OTHER = {20004: 'Other error.'}  # 其他问题

    def code(self):
        return tuple(self.value.keys())[0]

    def msg(self):
        return tuple(self.value.values())[0]

    def exception_msg(self):
        return str(self.code()) + ":" + self.msg()


if __name__ == '__main__':
    print(Status.SUCCESS.code())
    print(Status.SUCCESS.msg())
    print(Status.SUCCESS.exception_msg())
