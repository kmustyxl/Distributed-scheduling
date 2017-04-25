# -*- coding: utf-8 -*-
'''
分布式工厂

Factory：2 Machine：5 Job：30
''' 
import numpy as np
import random
import xlwt
from tkinter import *
from tkinter.messagebox import *
import matplotlib.pyplot as plt
book = xlwt.Workbook(encoding = 'utf-8', style_compression = 0)
sheet = book.add_sheet('data', cell_overwrite_ok = True)
global a, b

def get_machine():
    a = f1_m.get()
    b = f2_m.get()
    return a, b


def LoadData(filename):
    '''
    read the standard testing file
    '''
    fr = open(filename)
    dataMat = []
    temp = []
    num_machine = len(fr.readline().split(' ')) - 1
    a, b = get_machine()
    num_f1_machine = np.int(a)
    num_f2_machine = np.int(b)
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
def dataksh():
    plt.figure()
    plt.plot(x1,y1,c='red',label='$Facotoy1$')
    plt.plot(x2,y2,c='blue',label='$Facotoy2$')
    plt.xlabel('Individual')
    plt.ylabel('Fitness')
    plt.title('Distributed factory flow-shop by Python')
    plt.legend()
    plt.show()
    return

def GA(): 
    dataMat, num_f1_machine, num_f2_machine, num_job = LoadData('data\\30_5 dependent.txt')
    popsize = 2*num_job
    global x1, x2, y1, y2,factory1_job, factory2_job
    y1 = []
    y2 = []
    x1 = []
    x2 = []
    for gen in range(1):
        factory1_job, factory2_job, num_f1_job, num_f2_job = JobDistribute(num_job)
        c[0] = factory1_job
        d[0] = factory2_job
        frame3 = LabelFrame(text='工厂一分配机器号',font='24px',bd='2px',fg='#225A85')
        frame3.grid(row=3, sticky=W,padx=20)
        frame4 = LabelFrame(text='工厂二分配机器号',font='24px',bd='2px',fg='#225A85')
        frame4.grid(row=4, sticky=W,padx=20)
        lb2 = Listbox(frame3,height=1)
        for i in c:
            lb2.insert(END, i)
        lb2.pack()
        lb3 = Listbox(frame4,height=1)
        for i in d:
            lb3.insert(END, i)
        lb3.pack()
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
        frame1 = LabelFrame(text='工厂一输出区域',font='24px',bd='2px',fg='#225A85')
        frame1.grid(row=4, column=0, sticky=W,padx=20)       
        sb = Scrollbar(frame1)
        sb.pack(side=RIGHT, fill=Y)
        lb = Listbox(frame1, yscrollcommand=sb.set,height=20)
        for i in population1_mat:
            lb.insert(END, i)
        lb.pack(side=LEFT,fill=BOTH)

        lb00 = Listbox(frame1, yscrollcommand=sb.set,height=20)
        for i in t_fitness1:
            lb00.insert(END, i)
        lb00.pack(side=RIGHT,fill=BOTH)
        sb.config(command=lb.yview)
        sb.config(command=lb00.yview)
        
        frame2 = LabelFrame(text='工厂二输出区域',font='24px',bd='2px',fg='#225A85')
        frame2.grid(row=4, column=1)
        sb1 = Scrollbar(frame2)
        sb1.pack(side=RIGHT, fill=Y)
        lb1 = Listbox(frame2, yscrollcommand=sb1.set,height=20)
        for i in population2_mat:
            lb1.insert(END, i)
        lb1.pack(side=LEFT,fill=BOTH)
        lb11 = Listbox(frame2, yscrollcommand=sb1.set,height=20)
        for i in t_fitness2:
            lb11.insert(END, i)
        lb11.pack(side=LEFT,fill=BOTH)
        sb1.config(command=lb1.yview)
        sb1.config(command=lb11.yview)        
        
        for i in range(popsize): 
            for j in range(num_f1_job):
                sheet.write(i + popsize*gen, j, str(population1_mat[i][j])) 
                sheet.write(i + popsize*gen, num_f1_job, str(t_fitness1[i]))
                sheet.write(i + popsize*gen, num_f1_job + 1 + j, str(population2_mat[i][j]))
                sheet.write(i + popsize*gen, num_job+2, str(t_fitness2[i])) 
            y1.append(t_fitness1[i])
            y2.append(t_fitness2[i])
            x1.append(i + popsize*gen)
            x2.append(i + popsize*gen)
        frame0 = LabelFrame(text='数据可视化与管理',font='24px',bd='2px',fg='#225A85')
        frame0.grid(row=0, column=1,padx=20,pady=10)
        button11 = Button(frame0,text='数据可视化',width=10,command=dataksh)
        button11.grid(row=3, column=0)


    book.save('data\\data_caiji.xls')
    label22 = Label(frame0,text='数据采集状态：',fg='#225A85',pady='20')
    label22.grid(row=2, column=0)
    f22_m=StringVar()   
    f22_m.set('Success!!!')         
    txt22=Entry(frame0,textvariable=f22_m)
    txt22.grid(row=2, column=1) 

def APP():
    tf.destroy()
    global frame ,f1_m, f2_m, f22_m
    frame = LabelFrame(text='双工厂机器配置',font='24px',bd='2px',fg='#225A85')
    frame.grid(row=0, column=0,padx=20,pady=10)
    title = Label(frame,text='分布式工厂流水车间调度数据采集系统',font='24px',bd='2px',fg='#225A85' ) 
    title.grid(row=0, column=1)
    label1 = Label(frame,text='请输入工厂一机器数：',fg='#225A85',pady='20')
    label1.grid(row=1, column=0)
    f1_m=StringVar()            
    txt1=Entry(frame,textvariable=f1_m)
    txt1.grid(row=1, column=1)
    label2 = Label(frame,text='请输入工厂二机器数：',fg='#225A85',pady='20')
    label2.grid(row=2, column=0)
    f2_m=StringVar()            
    txt2=Entry(frame,textvariable=f2_m)
    txt2.grid(row=2, column=1)  
    button1 = Button(frame,text='传入系统',width=10,command=get_machine)
    button1.grid(row=2, column=2)
    button2 = Button(frame,text='开始执行',width=10, command=GA)
    button2.grid(row=1, column=2)
    frame0 = LabelFrame(text='数据可视化与管理',font='24px',bd='2px',fg='#225A85')
    frame0.grid(row=0, column=1,padx=20,pady=10)
    button11 = Button(frame0,text='数据可视化',width=10)
    button11.grid(row=3, column=0)
    label22 = Label(frame0,text='数据采集状态：',fg='#225A85',pady='20')
    label22.grid(row=2, column=0)
    f22_m=StringVar()   
    f22_m.set('Waiting...')         
    txt22=Entry(frame0,textvariable=f22_m)
    txt22.grid(row=2, column=1)  
    frame1 = LabelFrame(text='工厂一输出区域',font='24px',bd='2px',fg='#225A85')
    frame1.grid(row=4, column=0, sticky=W,padx=20)
    frame2 = LabelFrame(text='工厂二输出区域',font='24px',bd='2px',fg='#225A85')
    frame2.grid(row=4, column=1)
    frame3 = LabelFrame(text='工厂一分配机器号',font='24px',bd='2px',fg='#225A85')
    frame3.grid(row=1, sticky=W,padx=20)
    frame4 = LabelFrame(text='工厂二分配机器号',font='24px',bd='2px',fg='#225A85')
    frame4.grid(row=1, sticky=E,padx=20)
    lb2 = Listbox(frame3,height=1)
    for i in c:
        lb2.insert(END, i)
    lb2.pack()
    lb3 = Listbox(frame4,height=1)
    for i in d:
        lb3.insert(END, i)
    lb3.pack()
    sb = Scrollbar(frame1)
    sb.pack(side=RIGHT, fill=Y)
    lb = Listbox(frame1, yscrollcommand=sb.set,height=20)
    for i in ['统计完毕显示工件排序']:
        lb.insert(END, i)
    lb.pack(side=LEFT,fill=BOTH)

    lb00 = Listbox(frame1, yscrollcommand=sb.set,height=20)
    for i in ['统计完毕显示适配值']:
        lb00.insert(END, i)
    lb00.pack(side=RIGHT,fill=BOTH) 

    sb.config(command=lb.yview)
    sb.config(command=lb00.yview)
    sb1 = Scrollbar(frame2)
    sb1.pack(side=RIGHT, fill=Y)
    lb1 = Listbox(frame2, yscrollcommand=sb1.set,height=20)
    for i in ['统计完毕显示工件排序']:
        lb1.insert(END, i)
    lb1.pack(side=LEFT,fill=BOTH)
    lb11 = Listbox(frame2, yscrollcommand=sb1.set,height=20)
    for i in ['统计完毕显示适配值']:
        lb11.insert(END, i)
    lb11.pack(side=LEFT,fill=BOTH)
    sb1.config(command=lb1.yview)
    sb1.config(command=lb11.yview)

global tf, root
root = Tk()
#设置窗口的大小宽x高+偏移量
root.geometry('850x630+500+200')
#设置窗口标题
root.title('数据采集系统')
c=['统计完毕显示分配结果']
d=['统计完毕显示分配结果']
#登录界面
tf=LabelFrame(text='分布式工厂流水车间调度数据采集系统登录',font='24px',bd='2px',fg='#225A85')
tf.pack(anchor=CENTER,pady=220,ipadx=40)



Label(tf,text='管理员ID：',fg='#225A85',pady='20').grid(row=1,column=1)
userid=StringVar()            
txtuid=Entry(tf,textvariable=userid)
txtuid.grid(row=1,column=2)

Label(tf,text='密码：',fg='#225A85').grid(row=2,column=1,sticky=E)
password=StringVar()            
txtpwd=Entry(tf,textvariable=password,show='*')
txtpwd.grid(row=2,column=2)
btclear=Button(tf,text='登录',fg='#225A85',padx='10',command=APP)
btclear.grid(row=2,column=3)

root.mainloop()