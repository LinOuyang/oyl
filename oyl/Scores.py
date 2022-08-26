import numpy as np


class ScoreFuns:


    def __init__(self, obs, pre, threshold=0.1, axis=None, eps=0.001):
        """
        No explain not yet.

        """
        if obs.shape != pre.shape:
            raise ValueError('The shape of "obs" and "pre" do not match')
        
        self.n_nums = np.sum(np.ones_like(obs),axis=axis)
        self.mask = np.isnan(obs)
        
        obs_ = np.where(obs >= threshold, 1., 0.)
        pre_ = np.where(pre >= threshold, 1., 0.)
        obs_[self.mask] = np.nan
        pre_[self.mask] = np.nan
        obs_t, pre_t = 1-obs_, 1-pre_
        
        self.hits = np.nansum(obs_*pre_,axis=axis)
        self.false_alarms = np.nansum(obs_t*pre_,axis=axis)
        self.misses =  np.nansum(obs_*pre_t,axis=axis)
        self.true_negative =  np.nansum(obs_t*pre_t,axis=axis)
        self.eps = eps

    def pod(self):
        """
        命中率：命中数/（实际总的降水数）
        """
        return self.hits/(self.hits+self.misses+self.eps)
        
    def far(self):
        """
        空报率：误警数/（总预报的降水数）
        """
        return self.false_alarms/(self.hits+self.false_alarms+self.eps)
        
        
    def hss(self):
        num=2*(self.hits*self.true_negative - self.misses*self.false_alarms)
        den = (self.misses**2 + self.false_alarms**2 + 2*self.hits*self.true_negative
                   + (self.misses + self.false_alarms)*(self.hits + self.true_negative))
        return num/(den+self.eps)

    def acc(self):
    	
        return (self.hits+self.true_negative)/self.n_nums

    def ets(self):
        num = (self.hits + self.false_alarms) * (self.hits + self.misses)
        den = self.hits + self.misses + self.false_alarms + self.true_negative
        Dr = num / den
        return (self.hits - Dr) / (self.hits + self.misses + self.false_alarms - Dr+self.eps)

    def bias(self, vmax=2):
        """
        偏差评分(Bias score)主要用来衡量模式对某一量级降水的预报偏差, 该评分在数值上等于预报区域内满足某降水阈值的总格点数与对应实况降水总格点数的比值。
        是用来反映降水总体预报效果的检验方法。
        当BIAS>1时, 表示预报结果较实况而言偏湿;
        当BIAS < 1时, 表示预报结果较实况而言偏干;
        当BIAS=1时, 则表示预报偏差为0, 即预报技巧最高。

        由于BIAS评分主要是用于衡量预报区域内满足某降水阈值的预报技巧, 并不能衡量降水的准确率, 因此还需引入公平技巧评分(Equitable Threat Score, ETS)用于衡量对流尺度集合预报的预报效果。
        ETS评分表示在预报区域内满足某降水阈值的降水预报结果相对于满足同样降水阈值的随机预报的预报技巧, 因此该评分有效地去除了随机降水概率对评分的影响, 相对而言更加公平、客观。
        根据定义, 如果ETS>0时, 表示对于某量级降水来说预报有技巧; 如果ETS≤0时则表示预报没有技巧; 如果ETS=1则表示该预报为完美预报。
        """
        bia = (self.hits + self.false_alarms)/(self.hits + self.misses + 0.000001)
        bia[bia>vmax] = vmax
        return bia
        
    def ts(self):
        return self.hits/(self.hits + self.false_alarms+self.misses+self.eps)


    def get(self,*args):
        if isinstance(args[0],(tuple,list)):
            args = args[0]
        dic = dict(ts=self.ts, pod=self.pod, far=self.far,
                   hss=self.hss, acc=self.acc, ets=self.ets, bias=self.bias)
        return [dic.get(s.lower(),lambda x: None)() for s in args]


        

if __name__=='__main__':
    pass


























