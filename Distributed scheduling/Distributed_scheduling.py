# -*- coding:utf_8 -*-
'''
分布式工厂

Factory：2 Machine：5 Job：30
''' 
import numpy as np
import random
import xlwt
import matplotlib.pyplot as plt
book = xlwt.Workbook(encoding = 'utf-8', style_compression = 0)
sheet = book.add_sheet('data', cell_overwrite_ok = True)
def LoadData(filename):
    '''
    read the standard testing file
    '''
    fr = open(filename)
    dataMat = []
    temp = []
    num_machine = len(fr.readline().split(' ')) - 1
    num_f1_machine = int(input('Please input the number of machines in factory1: '))
    num_f2_machine = int(input('Please input the number of machines in factory2: '))
    num_job = 0
    fr.seek(0)      #回到文件第一行
    for line in fr.readlines():
        temp = [int(i) for i in list(line.strip().split(' '))]
        num_job += 1 
        dataMat.append(temp)
    return dataMat, num_f1_machine, num_f2_machine, num_job

def JobDistribute(num_job):
    '''
    obtain the job of the factory1 and factory2
    '''
    select_list = list(range(1,num_job+1))
    factory1_job = random.sample(select_list,int(num_job / 2)) 
    for i in range(int(num_job / 2)):
        if factory1_job[i] in select_list:
           select_list.remove(factory1_job[i])
    factory2_job = select_list
    num_f1_job = len(factory1_job)
    num_f2_job = len(factory2_job)
    return factory1_job, factory2_job, num_f1_job, num_f2_job

def InitPop(factory1_job, factory2_job,num_f1_job, num_f2_job, popsize = 200):
    '''
    population initialization
    '''
    population1_mat = np.zeros([popsize,num_f1_job],dtype = int)
    population2_mat = np.zeros([popsize,num_f2_job],dtype = int)
    for i in range(popsize):
        population1_mat[i] = random.sample(factory1_job,num_f1_job)
        population2_mat[i] = random.sample(factory2_job,num_f2_job)
    return population1_mat, population2_mat

def CalcFitness_f1(num_job, dataMat, factory1_job, num_f1_machine, population1_mat,\
                num_f1_job, popsize = 200):
    '''
    calculate the each fitness of the popsize in factory1 
    '''
    c_time1 = np.zeros([num_job, num_f1_machine])
    t_fitness1 =np.zeros([popsize, 1])
    for i in range(popsize):
        c_time1[population1_mat[i][0]-1][0] = dataMat[population1_mat[i][0]-1][0]
        for j in range(1,num_f1_job):
            c_time1[population1_mat[i][j]-1][0] = c_time1[population1_mat[i][j-1]-1][0] + dataMat[population1_mat[i][j]-1][0]
        for j in range(1, num_f1_machine):
            c_time1[population1_mat[i][0]-1][j] = c_time1[population1_mat[i][0]-1][j - 1] + dataMat[population1_mat[i][0]-1][j]
        for j in range(1, num_f1_job):
            for k in range(1, num_f1_machine):
                c_time1[population1_mat[i][j]-1][k] = dataMat[population1_mat[i][j]-1][k] \
                + max(c_time1[population1_mat[i][j - 1]-1][k], c_time1[population1_mat[i][j]-1][k - 1])
        t_fitness1[i][0] = c_time1[population1_mat[i][num_f1_job - 1]-1][num_f1_machine - 1]
    return t_fitness1

def CalcFitness_f2(num_job, dataMat, factory2_job,num_f2_machine, population2_mat,\
                   num_f2_job, popsize = 200):
    '''
    calculate the each fitness of the popsize in factory2 
    '''
    c_time2 = np.zeros([num_job, num_f2_machine])
    t_fitness2 =np.zeros([popsize, 1])
    for i in range(popsize):
        c_time2[population2_mat[i][0]-1][0] = dataMat[population2_mat[i][0]-1][0]
        for j in range(1,num_f2_job):
            c_time2[population2_mat[i][j]-1][0] = c_time2[population2_mat[i][j-1]-1][0] + dataMat[population2_mat[i][j]-1][0]
        for j in range(1, num_f2_machine):
            c_time2[population2_mat[i][0]-1][j] = c_time2[population2_mat[i][0]-1][j - 1] + dataMat[population2_mat[i][0]-1][j]
        for j in range(1, num_f2_job):
            for k in range(1, num_f2_machine):
                c_time2[population2_mat[i][j]-1][k] = dataMat[population2_mat[i][j]-1][k] \
                + max(c_time2[population2_mat[i][j - 1]-1][k], c_time2[population2_mat[i][j]-1][k - 1])
        t_fitness2[i][0] = c_time2[population2_mat[i][num_f2_job - 1]-1][num_f2_machine - 1]
    return  t_fitness2

def SelectParent(t_fitness1, t_fitness2, population1_mat, population2_mat):
    '''
    select the parent1 and parent2 in each factory
    '''
    min_t_fitness1 = min(t_fitness1[:,0])
    fitness1_list, fitness1_index = np.unique(np.array(t_fitness1), return_index = True) #对fitness进行排序（小到大），并返回原始数组的下标
    f1_parent1_num = fitness1_index[0]
    f1_parent2_num = fitness1_index[1]
    f1_parent1 = population1_mat[f1_parent1_num]
    f1_parent2 = population1_mat[f1_parent2_num]
    min_t_fitness2 = min(t_fitness2[:,0])
    fitness2_list, fitness2_index = np.unique(np.array(t_fitness2), return_index = True)
    f2_parent1_num = fitness2_index[0]
    f2_parent2_num = fitness2_index[1]
    f2_parent1 = population2_mat[f2_parent1_num]
    f2_parent2 = population2_mat[f2_parent2_num]    
    return f1_parent1, f1_parent2, f2_parent1, f2_parent2

def crossover_f1(f1_parent1, f1_parent2):
    num_f1_job = len(f1_parent1)
    child_1 = [lambda x:0 for x in range(num_f1_job)]
    f1_temp1 = random.randint(1,num_f1_job - 2)
    while True:
        f1_temp2 = random.randint(1,num_f1_job - 2)
        if f1_temp1 != f1_temp2:
            break
    rand_pos1 = max(f1_temp1, f1_temp2)
    rand_pos2 = min(f1_temp1, f1_temp2)
    for i in range(rand_pos2):
        child_1[i] = f1_parent1[i] 
    for i in range(num_f1_job - 1, rand_pos1 , -1):
        child_1[i] = f1_parent1[i]
    for i in range(num_f1_job):
        if f1_parent2[i] not in child_1 and rand_pos2 <= num_f1_job - 1:
            child_1[rand_pos2] = f1_parent2[i]
            rand_pos2  += 1
    return child_1
 
def mutation_f1(child_1):
    num_f1_job = len(child_1)
    temp1 = random.randint(1,num_f1_job - 1)
    while True:
        temp2 = random.randint(1,num_f1_job - 1)
        if abs(temp1 - temp2) >= 2:
            break
    temp_individual = child_1[:]
    rand_pos1 = max(temp1, temp2)
    rand_pos2 = min(temp1, temp2)
    child_1[rand_pos2] = child_1[rand_pos1]
    for i in range(rand_pos2 + 1, rand_pos1):
        child_1[i] = temp_individual[i - 1]
    return child_1

def compute_f1(t_fitness1):
    f1_sumfitness = 0
    length = t_fitness1.shape[0]
    for i in range(length):
        f1_sumfitness += t_fitness1[i][0]
    return f1_sumfitness

def select_f1(t_fitness1, f1_sumfitness):
    rand_p = random.random()
    while rand_p == 0:
        rand_p = random.random()
    i = 0; sum_p_select = 0
    while i <= t_fitness1.shape[0] - 1 and sum_p_select < rand_p:
        sum_p_select += t_fitness1[i] / f1_sumfitness
        i += 1
    return i - 1
  
dataMat, num_f1_machine, num_f2_machine, num_job = LoadData('data\\30_5 dependent.txt')
popsize = 2*num_job
y1 = []
y2 = []
x1 = []
x2 = []
for gen in range(2):
    factory1_job, factory2_job, num_f1_job, num_f2_job = JobDistribute(num_job)
    f1_newpop_mat = np.zeros([popsize,num_f1_job],dtype = int)
    f1_newpop_fitness1 =np.zeros([popsize, 1])
    f1_sort_pop = np.zeros([2 * popsize, num_f1_job], dtype = int)
    f1_sort_fit = np.zeros([2 * popsize, 1])
    pmutation = 0.05
    f2_newpop_mat = np.zeros([popsize,num_f2_job],dtype = int)
    f2_newpop_fitness2 =np.zeros([popsize, 1])
    f2_sort_pop = np.zeros([2 * popsize, num_f2_job], dtype = int)
    f2_sort_fit = np.zeros([2 * popsize, 1])
    population1_mat, population2_mat =InitPop(factory1_job,factory2_job,num_f1_job, num_f2_job,  popsize)
    t_fitness1=CalcFitness_f1(num_job, dataMat, factory1_job, num_f1_machine, population1_mat, num_f1_job, popsize)
    t_fitness2=CalcFitness_f2(num_job, dataMat, factory2_job, num_f2_machine, population2_mat, num_f2_job, popsize) 
    f1_parent1, f1_parent2, f2_parent1, f2_parent2 = SelectParent(t_fitness1, t_fitness2, population1_mat, population2_mat)
    f1_sumfitness = compute_f1(t_fitness1)
    f2_sumfitness = compute_f1(t_fitness2)
    for i in range(popsize):
        mate1 =  select_f1(t_fitness1, f1_sumfitness)
        mate2 =  select_f1(t_fitness1, f1_sumfitness)
        f1_newpop_mat[i] = crossover_f1(population1_mat[mate1], population1_mat[mate2])
        temp1_rand = random.random()
        if temp1_rand <= pmutation:
            f1_newpop_mat[i] = mutation_f1(f1_newpop_mat[i]) 
    f1_newpop_fitness1 = CalcFitness_f1(num_job, dataMat, factory1_job, num_f1_machine, f1_newpop_mat,num_f1_job, popsize) 
    for i in range(popsize):
        f1_sort_pop[i][:] = population1_mat[i][:]
        f1_sort_pop[popsize + i][:] = f1_newpop_mat[i][:]
    for i in range(popsize):
        f1_sort_fit[i][:] = t_fitness1[i][:]
        f1_sort_fit[popsize + i][:] = f1_newpop_fitness1[i][:]
    temp_list = []
    for i in range(2*popsize):
        temp_list.append(f1_sort_fit[i][0])
    f1_new_popindex = np.argsort(temp_list)      #返回原序列标签
    for i in range(popsize):
        temp = f1_new_popindex[i]
        population1_mat[i][:] = f1_sort_pop[temp][:]
        t_fitness1[i][0] = f1_sort_fit[temp][0]
    for i in range(popsize):
        mate3 =  select_f1(t_fitness2, f2_sumfitness)
        mate4 =  select_f1(t_fitness2, f2_sumfitness)
        f2_newpop_mat[i] = crossover_f1(population2_mat[mate3], population2_mat[mate4])
        temp_rand = random.random()
        if temp_rand <= pmutation: 
            f2_newpop_mat[i] = mutation_f1(f2_newpop_mat[i]) 
    f2_newpop_fitness2 = CalcFitness_f2(num_job, dataMat, factory2_job, num_f2_machine, f2_newpop_mat,num_f2_job, popsize) 
    for i in range(popsize):
        f2_sort_pop[i][:] = population2_mat[i][:]
        f2_sort_pop[popsize + i][:] = f2_newpop_mat[i][:]
        #print(f1_sort_pop[popsize + i], popsize+i)
    for i in range(popsize):
        f2_sort_fit[i][:] = t_fitness2[i][:]
        f2_sort_fit[popsize + i][:] = f2_newpop_fitness2[i][:]
    temp_list = []
    for i in range(2*popsize):
        temp_list.append(f2_sort_fit[i][0])
    f2_new_popindex = np.argsort(temp_list)      #返回原序列标签
    for i in range(popsize):
        temp = f2_new_popindex[i]
        population2_mat[i][:] = f2_sort_pop[temp][:]
        t_fitness2[i][0] = f2_sort_fit[temp][0]
    for i in range(popsize):
        print(population1_mat[i], t_fitness1[i], population2_mat[i], t_fitness2[i])
        for j in range(num_f1_job):
            sheet.write(i + popsize*gen, j, str(population1_mat[i][j])) 
            sheet.write(i + popsize*gen, num_f1_job, str(t_fitness1[i]))
            sheet.write(i + popsize*gen, num_f1_job + 1 + j, str(population2_mat[i][j]))
            sheet.write(i + popsize*gen, num_job+2, str(t_fitness2[i])) 
        y1.append(t_fitness1[i])
        y2.append(t_fitness2[i])
        x1.append(i + popsize*gen)
        x2.append(i + popsize*gen)
plt.figure()
plt.plot(x1,y1,c='red',label='$Facotoy1$')
plt.plot(x2,y2,c='blue',label='$Facotoy2$')
plt.xlabel('Individual')
plt.ylabel('Fitness')
plt.title('Distributed factory flow-shop by Python')
plt.legend()
plt.show()
book.save('data\\data_caiji.xls')
print('Successfully gain the data')
from tkinter import *   
def about():  
    label = Label(root, text='王小涛_同學\n QQ:*********', fg='red', bg='black')  
    label.pack(expand=YES, fill=BOTH)  
root = Tk()  
menubar = Menu(root)  
filemenu = Menu(menubar, tearoff=0)  
filemenu.add_command(label='打开', command=hello)  
filemenu.add_command(label='保存')  
filemenu.add_separator()  
filemenu.add_command(label='退出', command=root.quit)  
menubar.add_cascade(label='文件', menu=filemenu)  
helpmenu = Menu(menubar, tearoff=0)  
helpmenu.add_command(label='关于作者', command=about)  
menubar.add_cascade(label='关于', menu=helpmenu)  
root.config(menu=menubar)  
root.geometry('200x400')  
root.mainloop() 



import numpy as np
from tkinter import *
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
  
#----------------------------------------------------------------------
def drawPic():
    """
    获取GUI界面设置的参数，利用该参数绘制图片
    """
     
    #获取GUI界面上的参数
    try:sampleCount=int(inputEntry.get())
    except:
        sampleCount=50
        print ('请输入整数')
        inputEntry.delete(0,END)
        inputEntry.insert(0,'50')
     
    #清空图像，以使得前后两次绘制的图像不会重叠
    drawPic.f.clf()
    drawPic.a=drawPic.f.add_subplot(111)
     
    #在[0,100]范围内随机生成sampleCount个数据点
    x=np.random.randint(0,100,size=sampleCount)
    y=np.random.randint(0,100,size=sampleCount)
    color=['b','r','y','g']
     
    #绘制这些随机点的散点图，颜色随机选取
    drawPic.a.scatter(x,y,s=3,color=color[np.random.randint(len(color))])
    drawPic.a.set_title('Demo: Draw N Random Dot')
    drawPic.canvas.show()
     
     
if __name__ == '__main__':
     
    matplotlib.use('TkAgg')
    root=Tk()
     
    #在Tk的GUI上放置一个画布，并用.grid()来调整布局
    drawPic.f = Figure(figsize=(5,4), dpi=100) 
    drawPic.canvas = FigureCanvasTkAgg(drawPic.f, master=root)
    drawPic.canvas.show()
    drawPic.canvas.get_tk_widget().grid(row=0, columnspan=3)    
  
    #放置标签、文本框和按钮等部件，并设置文本框的默认值和按钮的事件函数
    Label(root,text='请输入样本数量：').grid(row=1,column=0)
    inputEntry=Entry(root)
    inputEntry.grid(row=1,column=1)
    inputEntry.insert(0,'50')
    Button(root,text='画图',command=drawPic).grid(row=1,column=2,columnspan=3)
     
    #启动事件循环
    root.mainloop() 