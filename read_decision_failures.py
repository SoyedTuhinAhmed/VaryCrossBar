import torch
import time


class RefVarFi:
    def __init__(self, wl, mu_p, mu_ap, sigma_p, sigma_ap, mew_mul=1., std_mul=1.):
        self.cdf_wls = {}  # LUT
        self.wl = wl
        self.flip_prob = [0] * (2 * self.wl + 1)
        self.refs_baseline = {}
        self.cur_sum_means, self.aps_means = {}, {}
        self.base_refs_trunc = {}
        self.linear_step = 0
        self.start, self.end = 0, 0
        self.std_mul = std_mul
        self.mew_mul = mew_mul
        self.mu_p = mu_p
        self.mu_ap = mu_ap
        self.sigma_p = sigma_p
        self.sigma_ap = sigma_ap
        # print('FI init --> std_mul: ', self.std_mul, ' mew_mul ', self.mew_mul)
        # print('FI init --> mu_p: ', self.mu_p, ' sigma_p ', self.sigma_p)

    def gen_norm_ap_conductance(self, mean_ap=25.3997 * 1., std_ap=0.9318 * 1):  # 12.414

        """
        assumptions: temperature is 25C , operation voltage is 100mv
        :param mean_ap: mean resistance of ap state
        :param std_ap: sigma of ap state
        :return: normal distribution of p and ap state based on mean and std of each state
        """

        norm_ap = torch.distributions.normal.Normal(loc=mean_ap, scale=std_ap).sample([10000])
        norm_Gap = 1 / norm_ap
        mean_ap, std_ap = torch.mean(norm_Gap), torch.std(norm_Gap)
        return torch.Tensor([self.mu_ap * 1, self.sigma_ap * self.std_mul])  # 1T1R 25 deg

    def gen_norm_p_conductance(self, mean_p=18.5433 * 1., std_p=0.6198 * 1):  #

        """
        assumptions: temperature is 25C , operation voltage is 100mv
        :param mean_p: mean resistance of p state
        :param std_p: sigma of p state
        :return: normal distribution of p and ap state based on mean and std of each state
        """

        norm_p = torch.distributions.normal.Normal(loc=mean_p, scale=std_p).sample([10000])
        norm_Gp = 1 / norm_p
        mean_p, std_p = torch.mean(norm_Gp), torch.std(norm_Gp)
        return torch.Tensor([self.mu_p * self.mew_mul, self.sigma_p * self.std_mul])  # 1T1R 25 deg

    def gen_activation_distributions(self, wls=16, v_one=600):
        """
        Generate activation flipping probability from the possible combinations of LRS and HRS from number of wordlines activated
        :param wls: number of wordlines activated concurrently
        :param ref: sensing reference
        :param v_one:
        :return:
        """
        with torch.cuda.amp.autocast():
            # total_cells = [wl]*wl
            comb_ap = [i for i in range(wls + 1)]  # how many possible ap's per current sum
            for aps in comb_ap:
                ps = wls - aps
                cur_sum = aps * (-1) + ps
                if aps == 0:
                    rand_cond = torch.normal(mean=self.mu_p * self.mew_mul, std=self.sigma_p * self.std_mul, size=(1, ps), device="cuda", dtype=torch.half).mul(
                        v_one)  # random sample p number of conductance and convert to current
                    self.cur_sum_means[cur_sum] = rand_cond.sum()  # sum up the per cell currents and store in a look up table
                    """calculate end range of possible conductance distribution"""
                    ps_cdf = self.gen_norm_p_conductance() * v_one
                    self.end = ps_cdf[0].mul(ps).add(ps_cdf[1].pow(2).mul(ps).sqrt().mul(3))
                    del ps_cdf, rand_cond
                elif ps == 0:
                    rand_cond = torch.normal(mean=self.mu_ap * 1, std=self.sigma_ap * self.std_mul, size=(1, aps), device="cuda", dtype=torch.half).mul(
                        v_one)  # random sample p number of conductance and convert to current
                    self.cur_sum_means[cur_sum] = rand_cond.sum()  # sum up the per cell currents and store in a look up table
                    """calculate start range of possible conductance distribution"""
                    aps_cdf = self.gen_norm_ap_conductance() * v_one  # convert to current
                    self.start = aps_cdf[0].mul(aps).sub(aps_cdf[1].pow(2).mul(aps).sqrt().mul(3))
                    del aps_cdf, rand_cond
                else:
                    rand_cond_ps = torch.normal(mean=self.mu_p * self.mew_mul, std=self.sigma_p * self.std_mul, size=(1, ps), device="cuda",
                                                dtype=torch.half).mul(v_one)  # random sample p number of conductance and convert to current
                    rand_cond_aps = torch.normal(mean=self.mu_ap * 1, std=self.sigma_ap * self.std_mul, size=(1, aps), device="cuda", dtype=torch.half).mul(
                        v_one)  # random sample p number of conductance and convert to current
                    self.cur_sum_means[cur_sum] = rand_cond_ps.sum().add(rand_cond_aps.sum())
                    del rand_cond_ps, rand_cond_aps

    def ref_replace_act_rand(self, act, wls):
        """

        :param wls:
        :param act:
        :return:
        """
        wls = float(wls)
        self.wl = wls
        refs_values = torch.tensor(list(self.refs_baseline.values()), device="cuda")
        possible_acts = torch.arange(0, wls + 1, 2, device="cuda", dtype=torch.half)
        adc_acts = torch.tensor(list(self.refs_baseline.keys()), device="cuda", dtype=torch.half)
        rand_act = self.cur_sum_means[0]
        print("rand_act ", rand_act)
        act[act == 0] = adc_acts[torch.where(torch.ones(1).cuda() == rand_act.sub(refs_values).sign())[0][0]]
        print("possible_acts ", int(adc_acts.size(0) / 2))
        for inter_act in possible_acts:  # replace act with flip prob
            """ ref fi """
            if inter_act != 0:
                ts = (act == inter_act).sum()
                if ts != 0:
                    with torch.cuda.amp.autocast():
                        rand_act = self.cur_sum_means[inter_act.item()]
                        act[act == inter_act] = adc_acts[torch.where(torch.ones(1).cuda() == rand_act.sub(refs_values).sign())[0][0]]
                min_ts = (act == inter_act.mul(-1)).sum()
                if min_ts != 0:
                    with torch.cuda.amp.autocast():
                        rand_act = self.cur_sum_means[-1 * inter_act.item()]
                        act[act == inter_act.mul(-1)] = adc_acts[torch.where(torch.ones(1).cuda() == rand_act.sub(refs_values).sign())[0][0]]

        return act

    def gen_ref_res_baseline(self, wls):
        """

        :param wls:
        :return:
        """
        # self.start = self.cur_sum_means[-wls][0].sub(self.cur_sum_means[-wls][1].mul(3))
        # self.end = self.cur_sum_means[wls][0].add(self.cur_sum_means[wls][1].mul(3))
        self.linear_step = (self.end - self.start) / ((wls * 2) + 1)
        print(self.linear_step, self.start, self.end)
        refs = torch.arange(self.end - self.linear_step, self.start, -self.linear_step)
        print("refs ... ", refs)
        count = wls

        for ref in refs:
            self.refs_baseline[count] = ref
            count -= 1
        # self.refs_baseline[wls] = self.refs_baseline[wls] + 25
        self.refs_baseline[-wls] = self.refs_baseline[-wls] - self.linear_step
        print("refs_baseline ", self.refs_baseline.keys(), self.refs_baseline.values())

    def ref_fi(self, acts, wls):

        """
        :param acts:
        :return:
        :param wls:
        """

        with torch.cuda.amp.autocast():
            acts = self.ref_replace_act_rand(act=acts, wls=wls)  # replace acts with rand acts
            assert acts.min() >= torch.tensor(-wls - 1, device='cuda', dtype=torch.float16)
            assert acts.max() <= torch.tensor(wls + 1, device='cuda', dtype=torch.float16)

            return acts


"""
test
"""
mu_p = 0.1357
mu_ap = 0.0532
sigma_ap = 0.0025
sigma_p = 0.0055

# torch.manual_seed(0)
start = time.time()
f = RefVarFi(8, mu_p, mu_ap, sigma_p, sigma_ap)
f.gen_activation_distributions(wls=8)
f.gen_ref_res_baseline(wls=8)
act = torch.randint(-8, 9, (1, 20), dtype=torch.half, device="cuda")
act = act - (act % 2)
print("Acts, ", act)
print(f.ref_replace_act_rand(act, 8))
end = time.time()
print(end - start)
Acts = torch.tensor([[0., -4., 2., 4., -4., 6., -6., 0., 2., 4., 2., 2., -8., 2., 6., -8., -2., 4., -8., 4.]])
"""     torch.tensor([[0., -4., 2., 4., -4., 6., -6., 0., 2., 4., 2., 2., -8., 2., 6., -8., -2., 4., -8., 4.]])
        torch.tensor([[0., -4., 2., 4., -4., 6., -6., 0., 2., 4., 2., 2., -8., 2., 6., -8., -2., 4., -8., 4.]])
        torch.tensor([[0., -4., 2., 4., -4., 6., -6., 0., 2., 4., 2., 2., -8., 2., 6., -8., -2., 4., -8., 4.]])
        # 10.17, 10.29,
        torch.tensor([[0., -4., 2., 4., -4., 6., -6., 0., 2., 4., 2., 2., -8., 2., 6., -8., -2., 4., -8., 4.]])
        torch.tensor([[0., -4., 2., 4., -4., 6., -6., 0., 2., 4., 2., 2., -8., 2., 6., -8., -2., 4., -8., 4.]])
"""
