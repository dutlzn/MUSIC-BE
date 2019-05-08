#函数里面修改全局变量
flag = 2
def test():
    global flag
    flag = 1 
flag = 3
def print_flag():
    global flag  #获取全局变量flag
    print(flag)
print_flag()
print(flag)
