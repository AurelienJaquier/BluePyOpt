"""StoppingCriteria class"""

"""
Copyright (c) 2016-2021, EPFL/Blue Brain Project

 This file is part of BluePyOpt <https://github.com/BlueBrain/BluePyOpt>

 This library is free software; you can redistribute it and/or modify it under
 the terms of the GNU Lesser General Public License version 3.0 as published
 by the Free Software Foundation.

 This library is distributed in the hope that it will be useful, but WITHOUT
 ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 details.

 You should have received a copy of the GNU Lesser General Public License
 along with this library; if not, write to the Free Software Foundation, Inc.,
 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
"""

# pylint: disable=R0912, R0914

import logging
import numpy

from collections import deque

import bluepyopt.stoppingCriteria

logger = logging.getLogger("__main__")


def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


class MaxNGen(bluepyopt.stoppingCriteria.StoppingCriteria):
    """Max ngen stopping criteria class"""

    name = "Max ngen"

    def __init__(self, max_ngen):
        """Constructor"""
        super(MaxNGen, self).__init__()
        self.max_ngen = max_ngen

    def check(self, kwargs):
        """Check if the maximum number of iteration is reached"""
        gen = kwargs.get("gen")
        if gen > self.max_ngen:
            self.criteria_met = True


class Stagnation(bluepyopt.stoppingCriteria.StoppingCriteria):
    """Stagnation stopping criteria class"""

    name = "Stagnation"

    def __init__(self, lambda_, problem_size):
        """Constructor"""
        super(Stagnation, self).__init__()

        self.lambda_ = lambda_
        self.problem_size = problem_size
        self.stagnation_iter = None

        self.best = []
        self.median = []

    def check(self, kwargs):
        """Check if the population stopped improving"""
        ngen = kwargs.get("gen")
        population = kwargs.get("population")
        fitness = [ind.fitness.reduce for ind in population]
        fitness.sort()

        # condition to avoid duplicates when re-starting
        if len(self.best) < ngen:
            self.best.append(fitness[0])
            self.median.append(fitness[int(round(len(fitness) / 2.0))])
        self.stagnation_iter = int(
            numpy.ceil(0.2 * ngen + 120 + 30.0 * self.problem_size
                       / self.lambda_)
        )

        cbest = len(self.best) > self.stagnation_iter
        cmed = len(self.median) > self.stagnation_iter
        cbest2 = numpy.median(self.best[-20:]) >= numpy.median(
            self.best[-self.stagnation_iter:-self.stagnation_iter + 20]
        )
        cmed2 = numpy.median(self.median[-20:]) >= numpy.median(
            self.median[-self.stagnation_iter:-self.stagnation_iter + 20]
        )
        if cbest and cmed and cbest2 and cmed2:
            # self.criteria_met = True
            with open("stopcrit.txt", "a") as f:
                f.write(f"Stagnation at gen {ngen}.\n")


class Stagnationv2(bluepyopt.stoppingCriteria.StoppingCriteria):
    """Stagnation stopping criteria class"""

    name = "Stagnationv2"

    def __init__(self, lambda_, problem_size):
        """Constructor"""
        super(Stagnationv2, self).__init__()

        self.lambda_ = lambda_
        self.problem_size = problem_size
        self.stagnation_iter = None

        self.best = []
        self.median = []

        # for plotting
        self.gens = []
        self.med_now = []
        self.med_bf = []
        self.best_now = []
        self.best_bf = []

    # def check(self, kwargs):
    #     """Check if the population stopped improving"""
    #     ngen = kwargs.get("gen")
    #     population = kwargs.get("population")
    #     fitness = [ind.fitness.reduce for ind in population]
    #     fitness.sort()

    #     self.best.append(fitness[0])
    #     self.median.append(fitness[int(round(len(fitness) / 2.0))])
    #     self.stagnation_iter = int(
    #         numpy.ceil(120 + 30.0 * self.problem_size
    #                    / self.lambda_)
    #     )
    #     i20 = int(ngen * 0.2) # sample of last 20%
    #     i30 = int(i20 * 0.3) # 30% of the sample

    #     cbest = len(self.best) > self.stagnation_iter
    #     cmed = len(self.median) > self.stagnation_iter
    #     # compare median of last 30 % of the sample vs median of first 30% of the sample
    #     cbest2 = numpy.median(self.best[-i30:]) >= numpy.median(
    #         self.best[-i20:-i20 + i30]
    #     )
    #     cmed2 = numpy.median(self.median[-i30:]) >= numpy.median(
    #         self.median[-i20:-i20 + i30]
    #     )

    #     if cbest and cmed and cbest2 and cmed2:
    #         self.criteria_met = True

    def check(self, kwargs):
        """Check if the population stopped improving"""
        ngen = kwargs.get("gen")
        self.stagnation_iter = int(
            numpy.ceil(120 + 30.0 * self.problem_size
                       / self.lambda_)
        )
        i20 = int(ngen * 0.2) # sample of last 20%
        i30 = int(i20 * 0.3) # 30% of the sample

        cbest = len(self.best[:ngen]) > self.stagnation_iter
        cmed = len(self.median[:ngen]) > self.stagnation_iter
        # compare median of last 30 % of the sample vs median of first 30% of the sample
        cbest2 = numpy.median(self.best[ngen-i30:ngen]) >= numpy.median(
            self.best[ngen-i20:ngen-i20 + i30]
        )
        cmed2 = numpy.median(self.median[ngen-i30:ngen]) >= numpy.median(
            self.median[ngen-i20:ngen-i20 + i30]
        )

        self.gens.append(ngen)
        self.best_now.append(numpy.median(self.best[ngen-i30:ngen]))
        self.best_bf.append(numpy.median(self.best[ngen-i20:ngen-i20 + i30]))
        self.med_now.append(numpy.median(self.median[ngen-i30:ngen]))
        self.med_bf.append(numpy.median(self.median[ngen-i20:ngen-i20 + i30]))
        # logger.info(len(self.best))
        # logger.info(self.stagnation_iter)
        # logger.info(numpy.median(self.best[-i30:]))
        # logger.info(numpy.median(self.best[-i20:-i20 + i30]))
        if cbest2 or cmed2:
            logger.info(ngen)
            logger.info(cbest2)
            logger.info(cmed2)
        if cbest and cmed and cbest2 and cmed2:
            logger.info(f"stop at {ngen}")
            self.criteria_met = True


class Stagnationv3(bluepyopt.stoppingCriteria.StoppingCriteria):
    """Stagnation stopping criteria class"""

    name = "Stagnationv3"

    def __init__(self, lambda_, problem_size, threshold=0.0001):
        """Constructor"""
        super(Stagnationv3, self).__init__()

        self.lambda_ = lambda_
        self.problem_size = problem_size
        self.stagnation_iter = int(
            numpy.ceil(120 + 30.0 * self.problem_size
                       / self.lambda_)
        )
        self.threshold = threshold

        self.best = []
        self.median = []

    def check(self, kwargs):
        """Check if the fitness does not improve reasonably fast"""
        ngen = kwargs.get("gen")
        population = kwargs.get("population")
        # fitness = [ind.fitness.reduce for ind in population]
        # fitness.sort()

        # self.best.append(fitness[0])
        # self.median.append(fitness[int(round(len(fitness) / 2.0))])

        i20 = int(ngen * 0.2) # sample of last 20%
        i30 = int(i20 * 0.3) # 30% of the sample

        cbest = len(self.best[:ngen]) > self.stagnation_iter
        cmed = len(self.median[:ngen]) > self.stagnation_iter
        # compare median of last 30 % of the sample vs median of first 30% of the sample
        cbest2 = numpy.median(self.best[ngen-i30:ngen]) * (1 + self.threshold * i20) > numpy.median(
            self.best[ngen-i20:ngen-i20 + i30]
        )
        cmed2 = numpy.median(self.median[ngen-i30:ngen]) * (1 + self.threshold * i20) > numpy.median(
            self.median[ngen-i20:ngen-i20 + i30]
        )

        if cbest and cmed and cbest2 and cmed2:
            self.criteria_met = True

class Stagnationv4(bluepyopt.stoppingCriteria.StoppingCriteria):
    """Stagnation stopping criteria class"""

    name = "Stagnationv4"

    def __init__(self, lambda_, problem_size, threshold=0.0001):
        """Constructor"""
        super(Stagnationv4, self).__init__()

        self.lambda_ = lambda_
        self.problem_size = problem_size
        self.stagnation_iter = int(
            numpy.ceil(120 + 30.0 * self.problem_size
                       / self.lambda_)
        )
        self.threshold = threshold

        self.best = []

    def check(self, kwargs):
        """Check if the fitness of best model does not improve reasonably fast"""
        ngen = kwargs.get("gen")
        population = kwargs.get("population")
        # fitness = [ind.fitness.reduce for ind in population]
        # fitness.sort()

        # self.best.append(fitness[0])

        i20 = int(ngen * 0.2) # sample of last 20%
        i30 = int(i20 * 0.3) # 30% of the sample

        cbest = len(self.best[:ngen]) > self.stagnation_iter
        # compare median of last 30 % of the sample vs median of first 30% of the sample
        cbest2 = numpy.median(self.best[ngen-i30:ngen]) * (1 + self.threshold * i20) > numpy.median(
            self.best[ngen-i20:ngen-i20 + i30]
        )

        if cbest and cbest2:
            self.criteria_met = True

class Stagnationv5(bluepyopt.stoppingCriteria.StoppingCriteria):
    """Stagnation stopping criteria class"""

    name = "Stagnationv5"

    def __init__(self, lambda_, problem_size, threshold=0.01):
        """Constructor"""
        super(Stagnationv5, self).__init__()

        self.lambda_ = lambda_
        self.problem_size = problem_size
        self.stagnation_iter = int(
            numpy.ceil(120 + 30.0 * self.problem_size
                       / self.lambda_)
        )
        self.threshold = threshold

        self.best = []
        self.median = []

    def check(self, kwargs):
        """Check if the fitness does not improve over 1% over 100 gens"""
        ngen = kwargs.get("gen")
        population = kwargs.get("population")
        # fitness = [ind.fitness.reduce for ind in population]
        # fitness.sort()

        # self.best.append(fitness[0])
        # self.median.append(fitness[int(round(len(fitness) / 2.0))])

        cbest = len(self.best[:ngen]) > self.stagnation_iter
        cmed = len(self.median[:ngen]) > self.stagnation_iter
        # compare median of last 30 % of the sample vs median of first 30% of the sample
        cbest2 = numpy.median(self.best[ngen-20:ngen]) * (1 + self.threshold) > numpy.median(
            self.best[ngen-120:ngen-100]
        )
        cmed2 = numpy.median(self.median[ngen-20:ngen]) * (1 + self.threshold) > numpy.median(
            self.median[ngen-120:ngen-100]
        )

        if cbest and cmed and cbest2 and cmed2:
            self.criteria_met = True

class Stagnationv6(bluepyopt.stoppingCriteria.StoppingCriteria):
    """Stagnation stopping criteria class"""

    name = "Stagnationv6"

    def __init__(self, lambda_, problem_size, threshold=0.01):
        """Constructor"""
        super(Stagnationv6, self).__init__()

        self.lambda_ = lambda_
        self.problem_size = problem_size
        self.stagnation_iter = int(
            numpy.ceil(120 + 30.0 * self.problem_size
                       / self.lambda_)
        )
        self.threshold = threshold

        self.best = []

    def check(self, kwargs):
        """Check if the best model fitness does not improve over 1% over 100 gens"""
        ngen = kwargs.get("gen")
        population = kwargs.get("population")
        # fitness = [ind.fitness.reduce for ind in population]
        # fitness.sort()

        # self.best.append(fitness[0])
        # self.median.append(fitness[int(round(len(fitness) / 2.0))])

        cbest = len(self.best[:ngen]) > self.stagnation_iter
        # compare median of last 30 % of the sample vs median of first 30% of the sample
        cbest2 = numpy.median(self.best[ngen-20:ngen]) * (1 + self.threshold) > numpy.median(
            self.best[ngen-120:ngen-100]
        )

        if cbest and cbest2:
            self.criteria_met = True


class Stagnationv7(bluepyopt.stoppingCriteria.StoppingCriteria):
    """Stagnation stopping criteria class"""

    name = "Stagnationv7"

    def __init__(self, lambda_, problem_size, threshold=0.01, std_threshold=0.02):
        """Constructor"""
        super(Stagnationv7, self).__init__()

        self.lambda_ = lambda_
        self.problem_size = problem_size
        self.stagnation_iter = int(
            numpy.ceil(120 + 30.0 * self.problem_size
                       / self.lambda_)
        )
        self.threshold = threshold
        self.std_threshold = std_threshold

        self.best = []
        self.median = []

    def check(self, kwargs):
        """Check if the best model fitness does not improve over 1% over 100 gens
            with small variations in best model fitness
        """
        ngen = kwargs.get("gen")
        population = kwargs.get("population")
        # fitness = [ind.fitness.reduce for ind in population]
        # fitness.sort()

        # self.best.append(fitness[0])
        # self.median.append(fitness[int(round(len(fitness) / 2.0))])

        cbest = len(self.best[:ngen]) > self.stagnation_iter
        # compare median of last 30 % of the sample vs median of first 30% of the sample
        cbest2 = numpy.median(self.best[ngen-20:ngen]) * (1 + self.threshold) > numpy.median(
            self.best[ngen-120:ngen-100]
        )
        cbest3 = numpy.std(self.best[ngen-20:ngen + 1]) < self.std_threshold * self.best[ngen]


        if cbest and cbest2 and cbest3:
            self.criteria_met = True


class Stagnationv8(bluepyopt.stoppingCriteria.StoppingCriteria):
    """Stagnation stopping criteria class"""

    name = "Stagnationv8"

    def __init__(self, lambda_, problem_size, threshold=0.0001, std_threshold=0.02):
        """Constructor"""
        super(Stagnationv8, self).__init__()

        self.lambda_ = lambda_
        self.problem_size = problem_size
        self.stagnation_iter = int(
            numpy.ceil(120 + 30.0 * self.problem_size
                       / self.lambda_)
        )
        self.threshold = threshold
        self.std_threshold = std_threshold

        self.best = []

    def check(self, kwargs):
        """Check if the fitness of best model does not improve reasonably fast"""
        ngen = kwargs.get("gen")
        population = kwargs.get("population")
        # fitness = [ind.fitness.reduce for ind in population]
        # fitness.sort()

        # self.best.append(fitness[0])

        i20 = int(ngen * 0.2) # sample of last 20%
        i30 = int(i20 * 0.3) # 30% of the sample

        cbest = len(self.best[:ngen]) > self.stagnation_iter
        # compare median of last 30 % of the sample vs median of first 30% of the sample
        cbest2 = numpy.median(self.best[ngen-i30:ngen]) * (1 + self.threshold * i20) > numpy.median(
            self.best[ngen-i20:ngen-i20 + i30]
        )
        cbest3 = numpy.std(self.best[ngen-i30:ngen + 1]) < self.std_threshold * self.best[ngen]

        if cbest and cbest2 and cbest3:
            self.criteria_met = True


class Stagnationv9(bluepyopt.stoppingCriteria.StoppingCriteria):
    """Stagnation stopping criteria class"""

    name = "Stagnationv9"

    def __init__(self, lambda_, problem_size, std_threshold=0.02):
        """Constructor"""
        super(Stagnationv9, self).__init__()

        self.lambda_ = lambda_
        self.problem_size = problem_size
        self.stagnation_iter = int(
            numpy.ceil(120 + 30.0 * self.problem_size
                       / self.lambda_)
        )
        self.std_threshold = std_threshold

        self.best = []

    def check(self, kwargs):
        """Check if the std of best model is below %age of best fitness value"""
        ngen = kwargs.get("gen")
        # population = kwargs.get("population")
        # fitness = [ind.fitness.reduce for ind in population]
        # fitness.sort()

        # self.best.append(fitness[0])

        i20 = int(ngen * 0.2) # sample of last 20%
        i30 = int(i20 * 0.3) # 30% of the sample

        cbest = len(self.best[:ngen]) > self.stagnation_iter

        cbest3 = numpy.std(self.best[ngen-i30:ngen + 1]) < self.std_threshold * self.best[ngen]

        if cbest  and cbest3:
            self.criteria_met = True


class Stagnationv10(bluepyopt.stoppingCriteria.StoppingCriteria):
    """Stagnation stopping criteria class"""

    name = "Stagnationv10"

    def __init__(self, lambda_, problem_size, threshold=0.0001, std_threshold=0.02):
        """Constructor"""
        super(Stagnationv10, self).__init__()

        self.lambda_ = lambda_
        self.problem_size = problem_size
        self.stagnation_iter = int(
            numpy.ceil(120 + 30.0 * self.problem_size
                       / self.lambda_)
        )
        self.threshold = threshold
        self.std_threshold = std_threshold

        self.best = []

    def check(self, kwargs):
        """Check if the fitness of best model does not improve reasonably fast"""
        ngen = kwargs.get("gen")
        population = kwargs.get("population")
        # fitness = [ind.fitness.reduce for ind in population]
        # fitness.sort()

        # self.best.append(fitness[0])

        # i20 = max(int(ngen * 0.2), 60) # sample of last 20%
        i20 = int(ngen * 0.2) # sample of last 20%
        i30 = int(i20 * 0.3) # 30% of the sample

        cbest = len(self.best[:ngen]) > self.stagnation_iter
        # compare median of last 30 % of the sample vs median of first 30% of the sample
        cbest2 = numpy.median(self.best[ngen-i30:ngen]) * (1 + self.threshold * i20) > numpy.median(
            self.best[ngen-i20:ngen-i20 + i30]
        )
        # cbest3 = numpy.std(self.best[ngen-i30:ngen + 1]) < self.std_threshold * self.best[ngen]
        cbest3 = numpy.std(self.best[ngen-i30:ngen + 1]) < numpy.std(self.best[ngen-i20:ngen-i20 + i30])  * (1 + self.threshold * i20)

        if cbest and cbest2 and cbest3:
            self.criteria_met = True


class TolHistFun(bluepyopt.stoppingCriteria.StoppingCriteria):
    """TolHistFun stopping criteria class"""

    name = "TolHistFun"

    def __init__(self, lambda_, problem_size):
        """Constructor"""
        super(TolHistFun, self).__init__()
        self.tolhistfun = 10 ** -12
        self.mins = deque(maxlen=10 + int(numpy.ceil(30.0 * problem_size
                                                     / lambda_)))

    def check(self, kwargs):
        """Check if the range of the best values is smaller than
        the threshold"""
        population = kwargs.get("population")
        self.mins.append(numpy.min([ind.fitness.reduce for ind in population]))
        # if max(self.mins) - min(self.mins) > self.tolhistfun:
        #     logger.info("TolHistFun diff:")
        #     logger.info(max(self.mins) - min(self.mins))

        if (
            len(self.mins) == self.mins.maxlen
            and max(self.mins) - min(self.mins) < self.tolhistfun
        ):
            self.criteria_met = True


class EqualFunVals(bluepyopt.stoppingCriteria.StoppingCriteria):
    """EqualFunVals stopping criteria class"""

    name = "EqualFunVals"

    def __init__(self, lambda_, problem_size):
        """Constructor"""
        super(EqualFunVals, self).__init__()
        self.problem_size = problem_size
        self.equalvals = float(problem_size) / 3.0
        self.equalvals_k = int(numpy.ceil(0.1 + lambda_ / 4.0))
        self.equalvalues = []
        self.equalvalues_2 = []
        self.equalvalues_3 = []
        self.equalvalues_4 = []

    def check(self, kwargs):
        """Check if in 1/3rd of the last problem_size iterations the best and
        k'th best solutions are equal"""
        ngen = kwargs.get("gen")
        population = kwargs.get("population")

        fitness = [ind.fitness.reduce for ind in population]
        fitness.sort()

        if isclose(fitness[0], fitness[-self.equalvals_k], rel_tol=1e-6):
            self.equalvalues.append(1)
        else:
            self.equalvalues.append(0)

        # new code for testing
        if not hasattr(self, "equalvalues_2"):
            self.equalvalues_2 = []
        if not hasattr(self, "equalvalues_3"):
            self.equalvalues_3 = []
        if not hasattr(self, "equalvalues_4"):
            self.equalvalues_4 = []

        if isclose(fitness[0], fitness[-self.equalvals_k], rel_tol=1e-5):
            self.equalvalues_2.append(1)
        else:
            self.equalvalues_2.append(0)

        if isclose(fitness[0], fitness[-self.equalvals_k], rel_tol=1e-4):
            self.equalvalues_3.append(1)
        else:
            self.equalvalues_3.append(0)

        if isclose(fitness[0], fitness[-self.equalvals_k], rel_tol=1e-3):
            self.equalvalues_4.append(1)
        else:
            self.equalvalues_4.append(0)

        sample_gen = 2075
        if ngen == sample_gen:
            logger.info(f"Values taken at gen {sample_gen}")
            logger.info("EqualFunVals (1e-6, 1e-5, 1e-4, 1e-3):")
            logger.info(sum(self.equalvalues[-self.problem_size:]))
            logger.info(sum(self.equalvalues_2[-self.problem_size:]))
            logger.info(sum(self.equalvalues_3[-self.problem_size:]))
            logger.info(sum(self.equalvalues_4[-self.problem_size:]))

        if (
            ngen > self.problem_size
            and sum(self.equalvalues[-self.problem_size:]) > self.equalvals
        ):
            self.criteria_met = True


class TolX(bluepyopt.stoppingCriteria.StoppingCriteria):
    """TolX stopping criteria class"""

    name = "TolX"

    def __init__(self):
        """Constructor"""
        super(TolX, self).__init__()
        self.tolx = 10 ** -12

    def check(self, kwargs):
        """Check if all components of pc and sqrt(diag(C)) are smaller than
        a threshold"""
        pc = kwargs.get("pc")
        C = kwargs.get("C")

        if all(pc < self.tolx) and all(numpy.sqrt(numpy.diag(C)) < self.tolx):
            self.criteria_met = True


class TolUpSigma(bluepyopt.stoppingCriteria.StoppingCriteria):
    """TolUpSigma stopping criteria class"""

    name = "TolUpSigma"

    def __init__(self, sigma0):
        """Constructor"""
        super(TolUpSigma, self).__init__()
        self.sigma0 = sigma0
        self.tolupsigma = 10 ** 20

    def check(self, kwargs):
        """Check if the sigma/sigma0 ratio is bigger than a threshold"""
        sigma = kwargs.get("sigma")
        diagD = kwargs.get("diagD")

        if sigma / self.sigma0 > float(diagD[-1] ** 2) * self.tolupsigma:
            self.criteria_met = True


class ConditionCov(bluepyopt.stoppingCriteria.StoppingCriteria):
    """ConditionCov stopping criteria class"""

    name = "ConditionCov"

    def __init__(self):
        """Constructor"""
        super(ConditionCov, self).__init__()
        self.conditioncov = 10 ** 14

    def check(self, kwargs):
        """Check if the condition number of the covariance matrix is
        too large"""
        cond = kwargs.get("cond")

        if cond > self.conditioncov:
            self.criteria_met = True


class NoEffectAxis(bluepyopt.stoppingCriteria.StoppingCriteria):
    """NoEffectAxis stopping criteria class"""

    name = "NoEffectAxis"

    def __init__(self, problem_size):
        """Constructor"""
        super(NoEffectAxis, self).__init__()
        self.conditioncov = 10 ** 14
        self.problem_size = problem_size

    def check(self, kwargs):
        """Check if the coordinate axis std is too low"""
        ngen = kwargs.get("gen")
        centroid = kwargs.get("centroid")
        sigma = kwargs.get("sigma")
        diagD = kwargs.get("diagD")
        B = kwargs.get("B")

        noeffectaxis_index = ngen % self.problem_size

        if all(
            centroid
            == centroid
            + 0.1 * sigma * diagD[-noeffectaxis_index] * B[-noeffectaxis_index]
        ):
            self.criteria_met = True


class NoEffectCoor(bluepyopt.stoppingCriteria.StoppingCriteria):
    """NoEffectCoor stopping criteria class"""

    name = "NoEffectCoor"

    def __init__(self):
        """Constructor"""
        super(NoEffectCoor, self).__init__()

    def check(self, kwargs):
        """Check if main axis std has no effect"""
        centroid = kwargs.get("centroid")
        sigma = kwargs.get("sigma")
        C = kwargs.get("C")

        if any(centroid == centroid + 0.2 * sigma * numpy.diag(C)):
            self.criteria_met = True
            