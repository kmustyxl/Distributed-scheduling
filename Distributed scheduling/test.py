import numpy as np
import random
import Distributed_scheduling as Ds
dataMat, num_f1_machine, num_f2_machine, num_job = Ds.LoadData('data\\30_5 dependent.txt')
factory1_job, factory2_job, num_f1_job, num_f2_job = Ds.JobDistribute(num_job)
population1_mat, population2_mat = Ds.InitPop(factory1_job,factory2_job,num_f1_job, num_f2_job,  100)
print(population2_mat)