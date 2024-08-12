'''
-*- coding: utf-8 -*-
@File  : timer.py.py
@author: wangxiang4@tal
@Time  : 2023/09/20 21:27
'''
import time


class Timer():
    '''
        usage:

        初始化时传入本次记时的项目名称，
        在每个记录点执行record()并传入计时点名称
        toString()会计算各阶段耗时并输出成字符串形式方便打印或者记录日志

        example:

        timer = Timer("detection")

        ** some preprocess operation.. **
        timer.record("preprocess")

        ** model inference.. **
        timer.record("model")

        ** postprocess operation.. **
        timer.record("postprocess")

        print(timer.toString())

    '''

    def __init__(self, name=""):
        self.name = name
        self.timer = []
        self.start_t = time.time()

    def record(self, name):
        self.timer.append((name, time.time()))

    def toString(self):
        str = "{} time (".format(self.name)
        if len(self.timer) == 0:
            str = ""
        elif len(self.timer) == 1:
            str += "total: %.4fs)" % (self.timer[0][1] - self.start_t)
        else:
            pre_t = self.start_t
            for mp in self.timer:
                str += "%s: %.4fs  " % (mp[0], mp[1] - pre_t)
                pre_t = mp[1]
            str += "total: %.4fs)" % (pre_t - self.start_t)
        return str


if __name__ == '__main__':
    timer = Timer("detection")

    t = 0
    for i in range(100000):
        t += i
    timer.record("add")

    # for i in range(10):
    #     t*=i
    # timer.record("mul and mul")

    print(timer.toString())

