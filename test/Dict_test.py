# 创建一个空字典
my_dict = {}

# 创建一个具有初始值的字典
my_dict = {'name': 'Alice', 'age': 25, 'city': 'New York'}

# 访问字典中的值
print(my_dict['name'])  # 输出: Alice
print(my_dict['age'])   # 输出: 25

# 修改字典中的值
my_dict['age'] = 26
print(my_dict['age'])   # 输出: 26

# 添加新键值对
my_dict['email'] = 'alice@example.com'
print(my_dict)
# 输出: {'name': 'Alice', 'age': 26, 'city': 'New York', 'email': 'alice@example.com'}

# 删除键值对
del my_dict['city']
print(my_dict)
# 输出: {'name': 'Alice', 'age': 26, 'email': 'alice@example.com'}